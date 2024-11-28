'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-23 15:04:44
Email: haimingzhang@link.cuhk.edu.cn
Description: The D2World occupancy world model.
'''
#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import mmcv
import os
import os.path as osp
import torch
from torch import nn
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import copy
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.bevformer.dense_heads.d2_modules import preprocess
from .bevformer import BEVFormer
from mmdet3d.models import builder
from projects.mmdet3d_plugin.bevformer.losses.lovasz_loss import lovasz_softmax
from mmengine.registry import MODELS
from ..utils import e2e_predictor_utils, eval_utils
from ..utils.occ_utils import occ_to_voxel


@DETECTORS.register_module()
class ViDARD2World(BEVFormer):
    def __init__(self,
                 history_len,
                 # Future predictions.
                 future_pred_head,
                 future_pred_frame_num,  # number of future prediction frames.
                 test_future_frame_num,  # number of future prediction frames when testing.

                 # BEV configurations.
                 point_cloud_range,
                 bev_h,
                 bev_w,

                 # Augmentations.
                 # A1. randomly drop current image (to enhance temporal feature.)
                 random_drop_image_rate=0.0,
                 # A2. add noise to previous_bev_queue.
                 random_drop_prev_rate=0.0,
                 random_drop_prev_start_idx=1,
                 random_drop_prev_end_idx=None,
                 # A3. grid mask augmentation.
                 grid_mask_image=True,
                 grid_mask_backbone_feat=False,
                 grid_mask_fpn_feat=False,
                 grid_mask_prev=False,
                 grid_mask_cfg=dict(
                     use_h=True,
                     use_w=True,
                     rotate=1,
                     offset=False,
                     ratio=0.5,
                     mode=1,
                     prob=0.7
                 ),

                 # Supervision.
                 supervise_all_future=True,

                 # Visualize point cloud.
                 _viz_pcd_flag=False,
                 _viz_pcd_path='dbg/pred_pcd',  # root/{prefix}

                 # Test server submission.
                 _submission=False,  # Flags for submission.
                 _submission_path='submission/model',  # root/{prefix}
                
                 # XWorld parameters
                 expansion=8,
                 num_classes=2,
                 patch_size=2,
                 load_pred_occ=False,
                 pred_occ_is_binary=None,
                 use_grid_sample=True,
                 use_autoreg_test=False,

                 *args,
                 **kwargs,):

        super().__init__(*args, **kwargs)
        self.future_pred_head = builder.build_head(future_pred_head)
        self.future_pred_frame_num = future_pred_frame_num
        self.test_future_frame_num = test_future_frame_num
        # if not predict any future,
        #  then only predict current frame.
        self.only_train_cur_frame = (future_pred_frame_num == 0)

        self.history_len = history_len

        self.point_cloud_range = point_cloud_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.num_classes = num_classes
        self.expansion = expansion
        self.class_embeds = nn.Embedding(num_classes, expansion)
        self.patch_size = patch_size

        self.use_grid_sample = use_grid_sample
        self.load_pred_occ = load_pred_occ
        self.pred_occ_is_binary = pred_occ_is_binary
        self.use_binary_occ = True if num_classes == 2 else False
        self.use_autoreg_test = use_autoreg_test
        
        out_dim = 32
        self.predicter = nn.Sequential(
                nn.Linear(expansion, out_dim*2),
                nn.Softplus(),
                nn.Linear(out_dim*2, 1),
            )

        # Augmentations.
        self.random_drop_image_rate = random_drop_image_rate
        self.random_drop_prev_rate = random_drop_prev_rate
        self.random_drop_prev_start_idx = random_drop_prev_start_idx
        self.random_drop_prev_end_idx = random_drop_prev_end_idx

        # Grid mask.
        self.grid_mask_image = grid_mask_image
        self.grid_mask_backbone_feat = grid_mask_backbone_feat
        self.grid_mask_fpn_feat = grid_mask_fpn_feat
        self.grid_mask_prev = grid_mask_prev
        self.grid_mask = GridMask(**grid_mask_cfg)

        # Training configurations.
        # randomly sample one future for loss computation?
        self.supervise_all_future = supervise_all_future

        self._viz_pcd_flag = _viz_pcd_flag
        self._viz_pcd_path = _viz_pcd_path
        self._submission = _submission
        self._submission_path = _submission_path

        # remove useless parameters.
        del self.future_pred_head.transformer
        del self.future_pred_head.bev_embedding
        del self.future_pred_head.prev_frame_embedding
        del self.future_pred_head.can_bus_mlp
        del self.future_pred_head.positional_encoding

    def preprocess(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x) # bs, F, H, W, D, c

        x = x.reshape(bs, F, H, W, D * self.expansion).permute(0, 1, 4, 2, 3)

        x = preprocess.reshape_patch(x, patch_size=self.patch_size)
        return x

    def post_process(self, x):
        x = preprocess.reshape_patch_back(x, self.patch_size)

        x = rearrange(x, 'b f (d c) h w -> b f h w d c', c=self.expansion)
        logits = self.predicter(x)
        return logits
    
    def grid_sample_inputs(self, batched_input_occs):
        num_frames = self.history_len

        # grid sample the original occupancy to the target pc range.
        # generate normalized grid
        device = batched_input_occs.device
        x_size = 200
        y_size = 200
        z_size = 16
        x = torch.linspace(-1.0, 1.0, x_size).view(-1, 1, 1).repeat(1, y_size, z_size).to(device)
        y = torch.linspace(-1.0, 1.0, y_size).view(1, -1, 1).repeat(x_size, 1, z_size).to(device)
        z = torch.linspace(-1.0, 1.0, z_size).view(1, 1, -1).repeat(x_size, y_size, 1).to(device)
        grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

        grid[..., 0] = grid[..., 0] * (51.2 / 50)
        grid[..., 1] = grid[..., 1] * (51.2 / 50)
        grid[..., 2] = grid[..., 2] - (4 / 16.0)

        # add flow to grid
        _batched_input_occs = rearrange(batched_input_occs, 'b f h w d -> (b f) () h w d')
        _batched_input_occs = _batched_input_occs + 1

        bs = _batched_input_occs.shape[0]
        grid = grid.unsqueeze(0).expand(bs, -1, -1, -1, -1)
        _batched_input_occs = F.grid_sample(_batched_input_occs.float(), 
                                            grid.flip(-1).float(), 
                                            mode='nearest', 
                                            padding_mode='zeros',
                                            align_corners=True)

        batched_input_occs = rearrange(_batched_input_occs.long(), '(b f) () h w d -> b f h w d', f=num_frames)
        batched_input_occs[batched_input_occs == 0] = 12
        batched_input_occs = batched_input_occs - 1
        return batched_input_occs
    
    def get_batched_inputs(self, input_occs):
        """From the input occupancy to the batched input occupancy.

        Args:
            input_occs (_type_): the input occupancy if a list with shape (F, N, 2)

        Returns:
            _type_: # (bs, F, H, W, D)
        """
        batched_input_occs = []
        for bs in range(len(input_occs)):
            next_occs = []
            cur_occ = input_occs[bs]
            for _occ in cur_occ:
                next_occs.append(occ_to_voxel(_occ))
            batched_input_occs.append(torch.stack(next_occs, 0))
        batched_input_occs = torch.stack(batched_input_occs, 0)  # (bs, F, H, W, D)
        return batched_input_occs
    
    def get_occ_inputs(self, input_occs):
        if self.training:
            # when training, we assume the input occupancy is always semantic occupancy
            batched_input_occs = self.get_batched_inputs(input_occs)

            if self.use_grid_sample:
                batched_input_occs = self.grid_sample_inputs(batched_input_occs)
            
            if self.use_binary_occ:
                ## convert the occupancy to binary occupancy
                batched_input_occs[batched_input_occs != 11] = 0
                batched_input_occs[batched_input_occs == 11] = 1
        else:
            if self.load_pred_occ:
                # if load the predicted occupancy by other methods, 
                # we assume the occupancy is already with the shape of (bs, F, H, W, D)
                batched_input_occs = input_occs.long()

                assert self.pred_occ_is_binary is not None, \
                    'You must specify the pred_occ_is_binary when loading the predicted occupancy.'
                
                if self.pred_occ_is_binary:
                    assert not self.use_grid_sample, 'The grid sample is not supported for binary occupancy for now.'
                    assert self.use_binary_occ, 'You can only use binary occupancy as inputs when the predicted occ is binary.'
                else:
                    if self.use_grid_sample:
                        batched_input_occs = self.grid_sample_inputs(batched_input_occs)
                    
                    if self.use_binary_occ:
                        ## convert the occupancy to binary occupancy
                        batched_input_occs[batched_input_occs != 11] = 0
                        batched_input_occs[batched_input_occs == 11] = 1
            else:
                batched_input_occs = self.get_batched_inputs(input_occs)

                if self.use_grid_sample:
                    batched_input_occs = self.grid_sample_inputs(batched_input_occs)
                if self.use_binary_occ:
                    ## convert the occupancy to binary occupancy
                    batched_input_occs[batched_input_occs != 11] = 0
                    batched_input_occs[batched_input_occs == 11] = 1
            
        return batched_input_occs
    
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      input_occs=None,
                      img_metas=None,
                      img=None,
                      gt_points=None,
                      img_depth=None,
                      img_mask=None,
                      **kwargs,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            gt_points (torch.Tensor optional): groundtruth point clouds for future
                frames with shape (x, x, x). Defaults to None.
                The 0-th frame represents current frame for reference.
        Returns:
            dict: Losses of different branches.
        """
        num_frames = self.history_len

        batched_input_occs = self.get_occ_inputs(input_occs)

        # Preprocess the historical occupancy
        x = self.preprocess(batched_input_occs)

        # C2. Check whether the frame has previous frames.
        prev_bev_exists_list = []
        prev_img_metas = copy.deepcopy(img_metas)

        img_metas = [each[num_frames-1] for each in img_metas]

        assert len(prev_img_metas) == 1, 'Only supports bs=1 for now.'
        for prev_img_meta in prev_img_metas:  # Loop batch.
            max_key = len(prev_img_meta) - 1
            prev_bev_exists = True
            for k in range(max_key, -1, -1):
                each = prev_img_meta[k]
                prev_bev_exists_list.append(prev_bev_exists)
                prev_bev_exists = prev_bev_exists and each['prev_bev_exists']
        prev_bev_exists_list = np.array(prev_bev_exists_list)[::-1]

        valid_frames = [0]

        if self.supervise_all_future:
            valid_frames.extend(list(range(self.future_pred_frame_num)))
        else:  # randomly select one future frame for computing loss to save memory cost.
            train_frame = np.random.choice(np.arange(self.future_pred_frame_num), 1)[0]
            valid_frames.append(train_frame)

        # forecasting the future occupancy
        next_bev_preds = self.future_pred_head.forward_head(x)
        next_bev_preds = self.post_process(next_bev_preds)  # (bs, F, X, Y, Z, c)
        next_bev_preds = rearrange(next_bev_preds, 'b f x y z c -> b f c y x z')
        next_bev_preds = rearrange(next_bev_preds, 'b f c y x z -> (b f) c () () (y x) z')

        # next_bev_preds = rearrange(next_bev_preds, 'b f h w d c -> b f (h w d) c')
        pred_dict = {
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
            'full_prev_bev_exists': prev_bev_exists_list.all(),
            'prev_bev_exists_list': prev_bev_exists_list,
        }

        # 5. Compute loss for point cloud predictions.
        start_idx = 0
        losses = dict()
        loss_dict = self.future_pred_head.loss(
            pred_dict, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range,
            pred_frame_num=self.future_pred_frame_num+1,
            img_metas=img_metas)
        losses.update(loss_dict)
        return losses

    def forward_test(self, 
                     img_metas, 
                     img=None,
                     gt_points=None, 
                     input_occs=None,
                     **kwargs):
        num_frames = self.history_len

        self.eval()

        batched_input_occs = self.get_occ_inputs(input_occs)

        # Preprocess the historical occupancy
        x = self.preprocess(batched_input_occs)

        next_bev_feats, valid_frames = [], []

        # 3. predict future BEV.
        valid_frames.extend(list(range(self.test_future_frame_num)))
        img_metas = [each[num_frames - 1] for each in img_metas]

        # forecasting the future occupancy
        next_bev_preds = self.future_pred_head.forward_head(x)
        next_bev_preds = self.post_process(next_bev_preds)  # (bs, F, X, Y, Z, c)
        next_bev_preds = rearrange(next_bev_preds, 'b f x y z c -> b f c y x z')
        next_bev_preds = rearrange(next_bev_preds, 'b f c y x z -> (b f) c () () (y x) z')

        pred_dict = {
            'next_bev_features': next_bev_feats,
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
        }

        # decode results and compute some statistic results if needed.
        start_idx = 0
        decode_dict = self.future_pred_head.get_point_cloud_prediction(
            pred_dict, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range, img_metas=img_metas)

        # convert decode_dict to quantitative statistics.
        pred_pcds = decode_dict['pred_pcds']
        gt_pcds = decode_dict['gt_pcds']
        scene_origin = decode_dict['origin']

        pred_frame_num = len(pred_pcds[0])
        ret_dict = dict()
        for frame_idx in range(pred_frame_num):
            count = 0
            frame_name = frame_idx + start_idx
            ret_dict[f'frame.{frame_name}'] = dict(
                count=0,
                chamfer_distance=0,
                l1_error=0,
                absrel_error=0,
            )
            for bs in range(len(pred_pcds)):
                pred_pcd = pred_pcds[bs][frame_idx]
                gt_pcd = gt_pcds[bs][frame_idx]

                ret_dict[f'frame.{frame_name}']['chamfer_distance'] += (
                    e2e_predictor_utils.compute_chamfer_distance_inner(
                        pred_pcd, gt_pcd, self.point_cloud_range).item())

                l1_error, absrel_error = eval_utils.compute_ray_errors(
                    pred_pcd.cpu().numpy(), gt_pcd.cpu().numpy(),
                    scene_origin[bs, frame_idx].cpu().numpy(), scene_origin.device)
                ret_dict[f'frame.{frame_name}']['l1_error'] += l1_error
                ret_dict[f'frame.{frame_name}']['absrel_error'] += absrel_error

                if self._viz_pcd_flag:
                    cur_name = img_metas[bs]['sample_idx']
                    out_path = f'{self._viz_pcd_path}_{cur_name}_{frame_name}.png'
                    gt_inside_mask = e2e_predictor_utils.get_inside_mask(gt_pcd, self.point_cloud_range)
                    gt_pcd_inside = gt_pcd[gt_inside_mask]
                    pred_pcd_inside = pred_pcd[gt_inside_mask]
                    root_path = '/'.join(out_path.split('/')[:-1])
                    mmcv.mkdir_or_exist(root_path)
                    self._viz_pcd(
                        pred_pcd_inside.cpu().numpy(),
                        scene_origin[bs, frame_idx].cpu().numpy()[None, :],
                        output_path=out_path,
                        gt_pcd=gt_pcd_inside.cpu().numpy()
                    )

                if self._submission:
                    self._save_prediction(pred_pcd, img_metas[bs], frame_idx + 1)

                count += 1
            ret_dict[f'frame.{frame_name}']['count'] = count

        if self._viz_pcd_flag:
            print('==== Visualize predicted point clouds done!! End the program. ====')
            print(f'==== The visualized point clouds are stored at {out_path} ====')
        return [ret_dict]

    
    def _save_prediction(self, pred_pcd, img_meta, frame_idx):
        """ Save prediction.

        The filename is <index>-<future-id>.txt
        In each line of the file: pred_depth
        """
        base_name = img_meta['sample_idx']
        base_name = f'{base_name}_{frame_idx}.txt'
        mmcv.mkdir_or_exist(self._submission_path)
        base_name = os.path.join(self._submission_path, base_name)

        r_depth = torch.sqrt((pred_pcd ** 2).sum(1)).cpu().numpy()

        with open(base_name, 'w') as f:
            for d in r_depth:
                f.write('%f\n' % (d))

    def _viz_pcd(self, pred_pcd, pred_ctr,  output_path, gt_pcd=None):
        """Visualize predicted future point cloud."""
        color_map = np.array([
            [0, 0, 230], [219, 112, 147], [255, 0, 0]
        ])
        pred_label = np.ones_like(pred_pcd)[:, 0].astype(np.int) * 0
        if gt_pcd is not None:
            gt_label = np.ones_like(gt_pcd)[:, 0].astype(np.int)

            pred_label = np.concatenate([pred_label, gt_label], 0)
            pred_pcd = np.concatenate([pred_pcd, gt_pcd], 0)

        e2e_predictor_utils._dbg_draw_pc_function(
            pred_pcd, pred_label, color_map, output_path=output_path,
            ctr=pred_ctr, ctr_labels=np.zeros_like(pred_ctr)[:, 0].astype(np.int)
        )


def warp_bev_features(voxel_feats, 
                      voxel_flow,
                      voxel_size, 
                      occ_size,
                      curr_ego_to_future_ego=None):
    """Warp the given voxel features using the predicted voxel flow.

    Args:
        voxel_feats (Tensor): _description_
        voxel_flow (Tensor): (bs, f, H, W, 2)
        voxel_size (Tensor): the voxel size for each voxel, for example torch.Tensor([0.4, 0.4])
        occ_size (Tensor): the size of the occupancy map, for example torch.Tensor([200, 200])
        extrinsic_matrix (_type_, optional): global to ego transformation matrix. Defaults to None.

    Returns:
        _type_: _description_
    """
    device = voxel_feats.device
    bs, num_pred, x_size, y_size, c = voxel_flow.shape

    if curr_ego_to_future_ego is not None:
        for i in range(bs):
            _extrinsic_matrix = curr_ego_to_future_ego[i]
            _voxel_flow = voxel_flow[i].reshape(num_pred, -1, 2)
            ## padding the zero flow for z axis
            _voxel_flow = torch.cat([_voxel_flow, torch.zeros(num_pred, _voxel_flow.shape[1], 1).to(device)], dim=-1)
            trans_flow = torch.matmul(_extrinsic_matrix[..., :3, :3], _voxel_flow.permute(0, 2, 1))
            trans_flow = trans_flow + _extrinsic_matrix[..., :3, 3][:, :, None]
            trans_flow = trans_flow.permute(0, 2, 1)[..., :2]
            voxel_flow[i] = trans_flow.reshape(num_pred, *voxel_flow.shape[2:])

    voxel_flow = rearrange(voxel_flow, 'b f h w dim2 -> (b f) h w dim2')
    new_bs = voxel_flow.shape[0]

    # normalize the flow in m/s unit to voxel unit and then to [-1, 1]
    voxel_size = voxel_size.to(device)
    occ_size = occ_size.to(device)

    voxel_flow = voxel_flow / voxel_size / occ_size

    # generate normalized grid
    x = torch.linspace(-1.0, 1.0, x_size).view(-1, 1).repeat(1, y_size).to(device)
    y = torch.linspace(-1.0, 1.0, y_size).view(1, -1).repeat(x_size, 1).to(device)
    grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)  # (h, w, 2)
    
    # add flow to grid
    grid = grid.unsqueeze(0).expand(new_bs, -1, -1, -1).flip(-1) + voxel_flow

    # perform the voxel feature warping
    voxel_feats = torch.repeat_interleave(voxel_feats, num_pred, dim=0)
    warped_voxel_feats = F.grid_sample(voxel_feats, 
                                       grid.float(), 
                                       mode='bilinear', 
                                       padding_mode='border')
    warped_voxel_feats = rearrange(warped_voxel_feats, '(b f) c h w -> b f c h w', b=bs)

    return warped_voxel_feats


@DETECTORS.register_module()
class ViDARD2WorldWithFlow(ViDARD2World):
    def __init__(self,
                 refine_decoder=None,
                 pred_abs_flow=False,
                 **kwargs):
        super().__init__(**kwargs)

        # the predicted flow is absolute flow defined in the world coordinate
        self.pred_abs_flow = pred_abs_flow

        self.flow_net = nn.Sequential(
            nn.Linear(128, 32 * 2),
            nn.Softplus(),
            nn.Linear(32 * 2, 2),
        )
        
        model_channel = 128
        channel = 8 * 16 * 2 * 2
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(model_channel, model_channel, 3, 2, 1, output_padding=1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(model_channel, channel, 1, 1, 0, bias=False),
        )

        # self.refine_decoder = builder.build_backbone(refine_decoder)
        self.refine_decoder = MODELS.build(refine_decoder)

        self.predicter2 = nn.Sequential(
            nn.Linear(self.expansion, 32*2),
            nn.Softplus(),
            nn.Linear(32*2, 1),
        )

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      input_occs=None,
                      img_metas=None,
                      img=None,
                      gt_points=None,
                      img_depth=None,
                      img_mask=None,
                      **kwargs,
                      ):
        num_frames = self.history_len

        batched_input_occs = self.get_occ_inputs(input_occs)

        # Preprocess the historical occupancy
        x = self.preprocess(batched_input_occs)

        # forecasting the future occupancy
        bev_flow_pred, input_feats = self.future_pred_head.forward_head(
            x, return_encoder_feat=True, **kwargs)
        bev_flow_pred = self.predict_bev_flow(bev_flow_pred)

        ## warp the current occupancy map using the predicted flow
        # bev_flow_pred = rearrange(bev_flow_pred, 'b f h w dim2 -> (b f) h w dim2')
        curr_bev_feat = self.extract_curr_bev_feat(input_feats)

        curr_ego_to_future_ego_trans = None
        if self.pred_abs_flow:
            metas = [each[num_frames-1] for each in img_metas]

            curr_ego_to_future_ego_list = []
            for meta in metas:
                curr_ego_to_future_ego = torch.from_numpy(meta['curr_ego_to_future_ego']).to(x.device)
                curr_ego_to_future_ego_list.append(curr_ego_to_future_ego.to(torch.float32))

            curr_ego_to_future_ego_trans = torch.stack(curr_ego_to_future_ego_list, dim=0)  # (bs, num_pred, 4, 4)
        
        if self.pred_abs_flow:
            warped_predicted_occ = warp_bev_features(
                curr_bev_feat, 
                bev_flow_pred, 
                voxel_size=torch.Tensor([51.2 * 2 / 200.0, 51.2 * 2 / 200.0]), 
                occ_size=torch.Tensor([200.0, 200.0]),
                curr_ego_to_future_ego=curr_ego_to_future_ego_trans)
        else:
            raise ValueError('Only support absolute flow prediction for now.')
            
        coarse_logits = self.get_coarse_occ(warped_predicted_occ)
        coarse_next_bev_preds = rearrange(coarse_logits, 'b f x y z c -> b f c y x z')
        coarse_next_bev_preds = rearrange(coarse_next_bev_preds, 'b f c y x z -> (b f) c () () (y x) z')

        refined_occ = self.refine_decoder(warped_predicted_occ)
        next_bev_preds = self.post_process2(refined_occ) # (bs, F, X, Y, Z, c)
        next_bev_preds = rearrange(next_bev_preds, 'b f x y z c -> b f c y x z')
        next_bev_preds = rearrange(next_bev_preds, 'b f c y x z -> (b f) c () () (y x) z')
        
        valid_frames = [0]
        if self.supervise_all_future:
            valid_frames.extend(list(range(self.future_pred_frame_num)))
        else:  # randomly select one future frame for computing loss to save memory cost.
            train_frame = np.random.choice(np.arange(self.future_pred_frame_num), 1)[0]
            valid_frames.append(train_frame)

        # C2. Check whether the frame has previous frames.
        prev_bev_exists_list = []
        prev_img_metas = copy.deepcopy(img_metas)

        img_metas = [each[num_frames-1] for each in img_metas]

        assert len(prev_img_metas) == 1, 'Only supports bs=1 for now.'
        for prev_img_meta in prev_img_metas:  # Loop batch.
            max_key = len(prev_img_meta) - 1
            prev_bev_exists = True
            for k in range(max_key, -1, -1):
                each = prev_img_meta[k]
                prev_bev_exists_list.append(prev_bev_exists)
                prev_bev_exists = prev_bev_exists and each['prev_bev_exists']
        prev_bev_exists_list = np.array(prev_bev_exists_list)[::-1]
        
        pred_dict = {
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
            'full_prev_bev_exists': prev_bev_exists_list.all(),
            'prev_bev_exists_list': prev_bev_exists_list,
        }

        # 5. Compute loss for point cloud predictions.
        start_idx = 0
        losses = dict()
        loss_dict = self.future_pred_head.loss(
            pred_dict, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range,
            pred_frame_num=self.future_pred_frame_num+1,
            img_metas=img_metas)
        losses.update(loss_dict)

        # compute the coarse loss
        coarse_pred_cit = {
            'next_bev_preds': coarse_next_bev_preds,
            'valid_frames': valid_frames,
            'full_prev_bev_exists': prev_bev_exists_list.all(),
            'prev_bev_exists_list': prev_bev_exists_list,
        }
        start_idx = 0
        coarse_loss_dict = self.future_pred_head.loss(
            coarse_pred_cit, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range,
            pred_frame_num=self.future_pred_frame_num+1,
            img_metas=img_metas, suffix='coarse')
        
        for key, value in coarse_loss_dict.items():
            coarse_loss_dict[key] = value * 0.5
        losses.update(coarse_loss_dict)

        return losses

    def get_coarse_occ(self, x):
        _x = rearrange(x, 'b f (d c) h w -> b f h w d c', c=self.expansion)
        logits = self.predicter(_x)
        return logits
    
    def predict_bev_flow(self, x):
        x = preprocess.reshape_patch_back(x, self.patch_size)

        x = rearrange(x, 'b f c h w -> b f h w c')
        logits = self.flow_net(x)
        return logits
    
    def extract_curr_bev_feat(self, x):
        x = x[:, -1]  # to (b, c, h, w)
        x = self.up_conv(x)[:, None]  # to (b, 1, c, h, w)

        x = preprocess.reshape_patch_back(x, self.patch_size)
        x = rearrange(x, 'b () c h w -> b c h w')
        return x
    
    def post_process2(self, x):
        x = rearrange(x, 'b f (d c) h w -> b f h w d c', c=self.expansion)
        logits = self.predicter2(x)
        return logits

    def forward_test(self, 
                     img_metas, 
                     img=None,
                     gt_points=None, 
                     input_occs=None,
                     **kwargs):
        if self.use_autoreg_test:
            return self.forward_test_autoreg(img_metas, 
                                             img, 
                                             gt_points, 
                                             input_occs, 
                                             **kwargs)
        
        num_frames = self.history_len

        self.eval()

        batched_input_occs = self.get_occ_inputs(input_occs)

        # Preprocess the historical occupancy
        x = self.preprocess(batched_input_occs)

        next_bev_feats, valid_frames = [], []

        # 3. predict future BEV.
        valid_frames.extend(list(range(self.test_future_frame_num)))

        # forecasting the future occupancy
        bev_flow_pred, input_feats = self.future_pred_head.forward_head(
            x, return_encoder_feat=True, **kwargs)
        bev_flow_pred = self.predict_bev_flow(bev_flow_pred)

        ## warp the current occupancy map using the predicted flow
        # bev_flow_pred = rearrange(bev_flow_pred, 'b f h w dim2 -> (b f) h w dim2')
        curr_bev_feat = self.extract_curr_bev_feat(input_feats)

        curr_ego_to_future_ego_trans = None
        if self.pred_abs_flow:
            metas = [each[num_frames-1] for each in img_metas]

            curr_ego_to_future_ego_list = []
            for meta in metas:
                curr_ego_to_future_ego = torch.from_numpy(meta['curr_ego_to_future_ego']).to(x.device)
                curr_ego_to_future_ego_list.append(curr_ego_to_future_ego.to(torch.float32))

            curr_ego_to_future_ego_trans = torch.stack(curr_ego_to_future_ego_list, dim=0)  # (bs, num_pred, 4, 4)
        
        if self.pred_abs_flow:
            warped_predicted_occ = warp_bev_features(
                curr_bev_feat, 
                bev_flow_pred, 
                voxel_size=torch.Tensor([51.2 * 2 / 200.0, 51.2 * 2 / 200.0]), 
                occ_size=torch.Tensor([200.0, 200.0]),
                curr_ego_to_future_ego=curr_ego_to_future_ego_trans)
        else:
            raise ValueError('Only support absolute flow prediction for now.')
            
        refined_occ = self.refine_decoder(warped_predicted_occ)

        next_bev_preds = self.post_process2(refined_occ)  # (bs, F, X, Y, Z, c)
        next_bev_preds = rearrange(next_bev_preds, 'b f x y z c -> b f c y x z')
        next_bev_preds = rearrange(next_bev_preds, 'b f c y x z -> (b f) c () () (y x) z')

        pred_dict = {
            'next_bev_features': next_bev_feats,
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
        }

        # from list to batched tensor
        img_metas = [each[num_frames - 1] for each in img_metas]

        # decode results and compute some statistic results if needed.
        start_idx = 0
        decode_dict = self.future_pred_head.get_point_cloud_prediction(
            pred_dict, gt_points, start_idx,
            tgt_bev_h=self.bev_h, tgt_bev_w=self.bev_w,
            tgt_pc_range=self.point_cloud_range, img_metas=img_metas)

        # convert decode_dict to quantitative statistics.
        pred_pcds = decode_dict['pred_pcds']
        gt_pcds = decode_dict['gt_pcds']
        scene_origin = decode_dict['origin']

        pred_frame_num = len(pred_pcds[0])
        ret_dict = dict()
        for frame_idx in range(pred_frame_num):
            count = 0
            frame_name = frame_idx + start_idx
            ret_dict[f'frame.{frame_name}'] = dict(
                count=0,
                chamfer_distance=0,
                l1_error=0,
                absrel_error=0,
            )
            for bs in range(len(pred_pcds)):
                pred_pcd = pred_pcds[bs][frame_idx]
                gt_pcd = gt_pcds[bs][frame_idx]

                ret_dict[f'frame.{frame_name}']['chamfer_distance'] += (
                    e2e_predictor_utils.compute_chamfer_distance_inner(
                        pred_pcd, gt_pcd, self.point_cloud_range).item())

                l1_error, absrel_error = eval_utils.compute_ray_errors(
                    pred_pcd.cpu().numpy(), gt_pcd.cpu().numpy(),
                    scene_origin[bs, frame_idx].cpu().numpy(), scene_origin.device)
                ret_dict[f'frame.{frame_name}']['l1_error'] += l1_error
                ret_dict[f'frame.{frame_name}']['absrel_error'] += absrel_error

                if self._viz_pcd_flag:
                    cur_name = img_metas[bs]['sample_idx']
                    out_path = f'{self._viz_pcd_path}_{cur_name}_{frame_name}.png'
                    gt_inside_mask = e2e_predictor_utils.get_inside_mask(gt_pcd, self.point_cloud_range)
                    gt_pcd_inside = gt_pcd[gt_inside_mask]
                    pred_pcd_inside = pred_pcd[gt_inside_mask]
                    root_path = '/'.join(out_path.split('/')[:-1])
                    mmcv.mkdir_or_exist(root_path)
                    self._viz_pcd(
                        pred_pcd_inside.cpu().numpy(),
                        scene_origin[bs, frame_idx].cpu().numpy()[None, :],
                        output_path=out_path,
                        gt_pcd=gt_pcd_inside.cpu().numpy()
                    )

                if self._submission:
                    self._save_prediction(pred_pcd, img_metas[bs], frame_idx + 1)

                count += 1
            ret_dict[f'frame.{frame_name}']['count'] = count

        if self._viz_pcd_flag:
            print('==== Visualize predicted point clouds done!! End the program. ====')
            print(f'==== The visualized point clouds are stored at {out_path} ====')
        return [ret_dict]
    


@DETECTORS.register_module()
class D2WorldOccWithFlow(ViDARD2WorldWithFlow):
    """Use XWorld to predict the occupancy rather than the point cloud

    Args:
        ViDARXWorldWithFlow (_type_): _description_
    """
    def __init__(self,
                 save_target_occs=False,
                 save_pred_dir=None,
                 **kwargs):
        super().__init__(**kwargs)

        out_dim = 32
        self.predicter = nn.Sequential(
            nn.Linear(self.expansion, out_dim*2),
            nn.Softplus(),
            nn.Linear(out_dim*2, self.num_classes),
        )
        self.predicter2 = nn.Sequential(
            nn.Linear(self.expansion, out_dim*2),
            nn.Softplus(),
            nn.Linear(out_dim*2, self.num_classes),
        )

        self.save_target_occs = save_target_occs
        self.save_pred_dir = save_pred_dir

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      input_occs=None,
                      img_metas=None,
                      img=None,
                      target_occs=None,
                      **kwargs,
                      ):
        num_frames = self.history_len

        batched_input_occs = self.get_occ_inputs(input_occs)

        # Preprocess the historical occupancy
        x = self.preprocess(batched_input_occs)

        # forecasting the future occupancy
        bev_flow_pred, input_feats = self.future_pred_head.forward_head(
            x, return_encoder_feat=True, **kwargs)
        bev_flow_pred = self.predict_bev_flow(bev_flow_pred)

        ## warp the current occupancy map using the predicted flow
        # bev_flow_pred = rearrange(bev_flow_pred, 'b f h w dim2 -> (b f) h w dim2')
        curr_bev_feat = self.extract_curr_bev_feat(input_feats)

        curr_ego_to_future_ego_trans = None
        if self.pred_abs_flow:
            metas = [each[num_frames-1] for each in img_metas]

            curr_ego_to_future_ego_list = []
            for meta in metas:
                curr_ego_to_future_ego = torch.from_numpy(meta['curr_ego_to_future_ego']).to(x.device)
                curr_ego_to_future_ego_list.append(curr_ego_to_future_ego.to(torch.float32))

            curr_ego_to_future_ego_trans = torch.stack(curr_ego_to_future_ego_list, dim=0)  # (bs, num_pred, 4, 4)
        
        if self.pred_abs_flow:
            warped_predicted_occ = warp_bev_features(
                curr_bev_feat, 
                bev_flow_pred, 
                voxel_size=torch.Tensor([50.0 * 2 / 200.0, 50.0 * 2 / 200.0]), 
                occ_size=torch.Tensor([200.0, 200.0]),
                curr_ego_to_future_ego=curr_ego_to_future_ego_trans)
        else:
            raise ValueError('Only support absolute flow prediction for now.')
        
        coarse_logits = self.get_coarse_occ(warped_predicted_occ)

        refined_occ = self.refine_decoder(warped_predicted_occ)
        refined_logits = self.post_process2(refined_occ) # (bs, F, X, Y, Z, c)

        ## compute the loss
        losses = dict()
        
        target_occs = self.get_batched_inputs(target_occs)

        coarse_loss_dict = self.compute_loss(
            coarse_logits, target_occs, suffix='coarse')
        for key, value in coarse_loss_dict.items():
            coarse_loss_dict[key] = value * 0.5
        losses.update(coarse_loss_dict)

        refine_loss = self.compute_loss(
            refined_logits, target_occs, suffix='refine')
        losses.update(refine_loss)
        return losses

    
    def compute_loss(self, pred_occ, target_occ, suffix=''):
        loss_dict = dict()

        # 1) compute the reconstruction loss
        rec_loss = 10.0 * F.cross_entropy(
            pred_occ.permute(0, 5, 1, 2, 3, 4), 
            target_occ)
        loss_dict[f'{suffix}rec_loss'] = rec_loss
        
        # 2) compute the LovaszLoss
        pred_occ = pred_occ.flatten(0, 1).permute(0, 4, 1, 2, 3).softmax(dim=1)
        target_occ = target_occ.flatten(0, 1)
        lovasz_loss = lovasz_softmax(pred_occ, target_occ)
        loss_dict[f'{suffix}lovasz_loss'] = lovasz_loss
        return loss_dict

    def forward_test(self,
                     input_occs=None,
                     img_metas=None,
                     img=None,
                     target_occs=None,
                     **kwargs,
                     ):
        num_frames = self.history_len

        self.eval()

        batched_input_occs = self.get_occ_inputs(input_occs)

        # Preprocess the historical occupancy
        x = self.preprocess(batched_input_occs)

        # forecasting the future occupancy
        bev_flow_pred, input_feats = self.future_pred_head.forward_head(
            x, return_encoder_feat=True, **kwargs)
        bev_flow_pred = self.predict_bev_flow(bev_flow_pred)

        ## warp the current occupancy map using the predicted flow
        # bev_flow_pred = rearrange(bev_flow_pred, 'b f h w dim2 -> (b f) h w dim2')
        curr_bev_feat = self.extract_curr_bev_feat(input_feats)

        curr_ego_to_future_ego_trans = None
        if self.pred_abs_flow:
            metas = [each[num_frames-1] for each in img_metas]

            curr_ego_to_future_ego_list = []
            for meta in metas:
                curr_ego_to_future_ego = torch.from_numpy(meta['curr_ego_to_future_ego']).to(x.device)
                curr_ego_to_future_ego_list.append(curr_ego_to_future_ego.to(torch.float32))

            curr_ego_to_future_ego_trans = torch.stack(curr_ego_to_future_ego_list, dim=0)  # (bs, num_pred, 4, 4)
        
        if self.pred_abs_flow:
            warped_predicted_occ = warp_bev_features(
                curr_bev_feat, 
                bev_flow_pred, 
                voxel_size=torch.Tensor([50.0 * 2 / 200.0, 50.0 * 2 / 200.0]), 
                occ_size=torch.Tensor([200.0, 200.0]),
                curr_ego_to_future_ego=curr_ego_to_future_ego_trans)
        else:
            raise ValueError('Only support absolute flow prediction for now.')
        
        refined_occ = self.refine_decoder(warped_predicted_occ)
        refined_logits = self.post_process2(refined_occ) # (bs, F, X, Y, Z, c)

        # save results to be (bs, F, X, Y, Z)
        pred_pcds = refined_logits.argmax(dim=-1).detach().cpu().numpy()

        img_metas = [each[num_frames - 1] for each in img_metas]

        # save the results

        if self.save_target_occs:
            target_occs = self.get_batched_inputs(target_occs).cpu().numpy()
        
        future_img_metas = kwargs['future_img_metas']
        pred_frame_num = len(pred_pcds[0])
        for frame_idx in range(pred_frame_num):
            for bs in range(len(pred_pcds)):
                pred_pcd = pred_pcds[bs][frame_idx]

                if self.save_target_occs:
                    target_occ = target_occs[bs][frame_idx]
                    self.save_occ_results("./results/occupancy_val_gt",
                                          target_occ, img_metas[bs], frame_idx)
                
                future_meta = future_img_metas[bs][frame_idx]
                self.save_occ_results(self.save_pred_dir,
                                      pred_pcd, future_meta, frame_idx,
                                      use_occ_path=True)
        return [pred_pcds]
    
    def save_occ_results(self, 
                         save_dir, 
                         pred_pcd, 
                         img_meta, 
                         frame_idx,
                         use_occ_path=False):
        if use_occ_path:
            occ_gt_path = img_meta['occ_gt_path']
            save_path = occ_gt_path.replace('dataset/openscene-v1.0', save_dir)
            save_path = save_path.replace('data/openscene-v1.0', save_dir)
            save_dir = osp.split(save_path)[0]
            mmcv.mkdir_or_exist(save_dir)
        else:
            base_name = img_meta['sample_idx']
            base_name = f'{base_name}_{frame_idx}'
            mmcv.mkdir_or_exist(save_dir)
            save_path = os.path.join(save_dir, base_name)

        np.savez_compressed(save_path, pred_pcd.astype(np.uint8))



