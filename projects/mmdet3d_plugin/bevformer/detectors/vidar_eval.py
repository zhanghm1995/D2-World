'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-17 15:50:48
Email: haimingzhang@link.cuhk.edu.cn
Description: Only for offline evaluation.
'''
#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import mmcv
import os
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import copy
import numpy as np
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

from .bevformer import BEVFormer
from mmdet3d.models import builder
from ..utils import e2e_predictor_utils, eval_utils


def occ_to_voxel(occ_data,
                 point_cloud_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0], 
                 occupancy_size=[0.5, 0.5, 0.5],
                 occupancy_classes=11):

    occ_data = occ_data.long()
    occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
    occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
    occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])

    voxel_num = occ_xdim * occ_ydim * occ_zdim

    gt_occupancy = (torch.ones(voxel_num, dtype=torch.long) * occupancy_classes).to(occ_data.device)
    gt_occupancy[occ_data[:, 0]] = occ_data[:, 1]

    gt_occupancy = gt_occupancy.reshape(occ_zdim, occ_ydim, occ_xdim)

    gt_occupancy = gt_occupancy.permute(2, 1, 0)
    return gt_occupancy


@DETECTORS.register_module()
class ViDAREval(BEVFormer):
    def __init__(self,
                 # Future predictions.
                 future_pred_head,
                 future_pred_frame_num,  # number of future prediction frames.
                 test_future_frame_num,  # number of future prediction frames when testing.
                 
                 queue_length,
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

                 *args,
                 **kwargs,):

        super().__init__(*args, **kwargs)
        self.future_pred_head = builder.build_head(future_pred_head)
        self.future_pred_frame_num = future_pred_frame_num
        self.test_future_frame_num = test_future_frame_num

        self.queue_length = queue_length
        # if not predict any future,
        #  then only predict current frame.
        self.only_train_cur_frame = (future_pred_frame_num == 0)

        self.point_cloud_range = point_cloud_range
        self.bev_h = bev_h
        self.bev_w = bev_w

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

        if self.only_train_cur_frame:
            # remove useless parameters.
            del self.future_pred_head.transformer
            del self.future_pred_head.bev_embedding
            del self.future_pred_head.prev_frame_embedding
            del self.future_pred_head.can_bus_mlp
            del self.future_pred_head.positional_encoding


    def forward_test(self, img_metas, img=None,
                     gt_points=None, **kwargs):
        """has similar implementation with train forward."""
        num_frames = self.queue_length + 1  # the historical frames and 1 current frame
        self.eval()

        # 2. Align previous frames to reference coordinates.
        valid_frames = []
        valid_frames.append(0)

        # 3. predict future BEV.
        valid_frames.extend(list(range(1, self.test_future_frame_num+1)))
        img_metas = [each[num_frames - 1] for each in img_metas]

        ## convert the raw occ_gts into to occupancy
        occ_gts_raw = kwargs['occ_gts']
        batched_next_occ = []
        for bs in range(len(occ_gts_raw)):
            next_occs = []
            cur_occ = occ_gts_raw[bs]
            for _occ in cur_occ:
                next_occs.append(occ_to_voxel(_occ))
            batched_next_occ.append(torch.stack(next_occs, 0))
        batched_next_occ = torch.stack(batched_next_occ, 0)

        # pred_frame_num, inter_num, bs, bev_h * bev_w, embed_dims.
        pred_dict = {
            'next_bev_preds': batched_next_occ,
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

                if self._submission and frame_idx > 0:
                    # ViDAR additionally predict the current frame as 0-th index.
                    #   So, we need to ignore the 0-th index by default.
                    self._save_prediction(pred_pcd, img_metas[bs], frame_idx)

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
