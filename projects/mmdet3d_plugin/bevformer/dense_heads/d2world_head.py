'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-23 15:41:12
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
<V1.multiframe> of ViDAR future prediction head:
    * Predict future & history frames simultaneously.
"""

import copy
import torch
import torch.nn as nn
import numpy as np

from mmdet.models import HEADS, build_loss

from mmcv.runner import force_fp32, auto_fp16
from .vidar_head_base import ViDARHeadBase

from .d2_modules.layers import EncoderLayer, DecoderLayer
from .mm_encoder import get_clones, MMEncoder



class Encoder(nn.Module):
    def __init__(self, in_channel, model_channel, N, heads):
        super().__init__()
        self.N = N
        self.model_channel = model_channel
        self.layers = get_clones(EncoderLayer(model_channel, heads), N)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, model_channel, 3, 2, 1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),

        )

    def forward(self, x, src_pos, src_ego_pose=None):
        bs, seq, c, h, w = x.size()
        x = self.conv(x.view(-1, *x.shape[2:]))
        x = x.view(bs, seq, self.model_channel, x.shape[-2], x.shape[-1])
        x = x + src_pos
        
        if src_ego_pose is not None:
            x = x + src_ego_pose

        for i in range(self.N):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, model_channel, channel, N, heads,
                 upsample_ratio=2):
        super().__init__()
        self.N = N
        self.model_channel = model_channel
        self.channel = channel
        self.layers = get_clones(DecoderLayer(model_channel, heads), N)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(model_channel, model_channel, 3, 2, 1, output_padding=1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(model_channel, channel, 1, 1, 0, bias=False),
        )

        self.upsample_ratio = upsample_ratio

    def forward(self, e_outputs, x, return_final_feat=False):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)  # (bs, seq, c, h, w)
            up_size = (x.size(3) * self.upsample_ratio, 
                       x.size(4) * self.upsample_ratio)

        x1 = self.conv(x.view(-1, *x.shape[2:])).view(x.size(0), x.size(1), 
                                                        self.channel, 
                                                        up_size[0], up_size[1])
        
        if return_final_feat:
            return x, x1
        return x1



@HEADS.register_module()
class D2WorldHead(ViDARHeadBase):
    def __init__(self,
                 history_queue_length,
                 d_model,
                 n_encoder_layers,
                 n_decoder_layers,
                 heads,
                 in_channel,
                 history_len=5,
                 future_len=6,
                 downsample_ratio=2,
                 pred_history_frame_num=0,
                 pred_future_frame_num=0,
                 per_frame_loss_weight=(1.0,),
                 use_occ_eval=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.history_queue_length = history_queue_length

        self.pred_history_frame_num = pred_history_frame_num
        self.pred_future_frame_num = pred_future_frame_num
        self.use_occ_eval = use_occ_eval

        self.pred_frame_num = 1 + self.pred_history_frame_num + self.pred_future_frame_num
        self.per_frame_loss_weight = per_frame_loss_weight
        assert len(self.per_frame_loss_weight) == self.pred_frame_num

        self.downsample_ratio = downsample_ratio
        self.encoder = Encoder(in_channel, d_model, n_encoder_layers, heads)
        self.decoder = Decoder(d_model, in_channel, n_decoder_layers, heads)
        self.pos_emb = nn.Embedding(history_len + future_len, d_model, max_norm=1, scale_grad_by_freq=True)

        self.d_model = d_model
        self.downsample_ratio = downsample_ratio
        self.history_len = history_len
        self.future_len = future_len
        self.total_len = history_len + future_len

    def forward_head(self, src, return_encoder_feat=False, **kwargs):
        """Forecast the future occupancy features according to the historical occupancy
        embeddings.

        Args:
            src (torch.Tensor): ()
        """
        down_size = (src.size(3) // self.downsample_ratio, 
                     src.size(4) // self.downsample_ratio)

        index = torch.arange(self.total_len, device=src.device)
        pos_emb = self.pos_emb(index).contiguous()
        pos_emb = pos_emb[None, :, :, None, None]
        pos_emb = pos_emb.expand(src.size(0), self.total_len, self.d_model, down_size[0], down_size[1])
        src_pos = pos_emb[:, :self.history_len]
        tgt_pos = pos_emb[:, self.history_len:]
        e_outputs = self.encoder(src, src_pos)
        d_output = self.decoder(e_outputs, tgt_pos)
        
        if return_encoder_feat:
            return d_output, e_outputs
        return d_output

    def _get_reference_gt_points(self,
                                 gt_points,
                                 src_frame_idx_list,
                                 tgt_frame_idx_list,
                                 img_metas):
        """Transform gt_points at src_frame_idx in {src_frame_idx_list} to the coordinate space
        of each tgt_frame_idx in {tgt_frame_idx_list}.
        """
        bs = len(gt_points)
        aligned_gt_points = []
        batched_origin_points = []
        for frame_idx, src_frame_idx, tgt_frame_idx in zip(
                range(len(src_frame_idx_list)), src_frame_idx_list, tgt_frame_idx_list):
            # 1. get gt_points belongs to src_frame_idx.
            src_frame_gt_points = [p[p[:, -1] == src_frame_idx] for p in gt_points]

            # 2. get transformation matrix..
            src_to_ref = [img_meta['total_cur2ref_lidar_transform'][src_frame_idx] for img_meta in img_metas]
            src_to_ref = gt_points[0].new_tensor(np.array(src_to_ref))  # bs, 4, 4
            ref_to_tgt = [img_meta['total_ref2cur_lidar_transform'][tgt_frame_idx] for img_meta in img_metas]
            ref_to_tgt = gt_points[0].new_tensor(np.array(ref_to_tgt))  # bs, 4, 4
            src_to_tgt = torch.matmul(src_to_ref, ref_to_tgt)

            # 3. transfer src_frame_gt_points to src_to_tgt.
            aligned_gt_points_per_frame = []
            for batch_idx, points in enumerate(src_frame_gt_points):
                new_points = points.clone()  # -1, 4
                new_points = torch.cat([
                    new_points[:, :3], new_points.new_ones(new_points.shape[0], 1)
                ], 1)
                new_points = torch.matmul(new_points, src_to_tgt[batch_idx])
                new_points[..., -1] = frame_idx
                aligned_gt_points_per_frame.append(new_points)
            aligned_gt_points.append(aligned_gt_points_per_frame)

            # 4. obtain the aligned origin points.
            aligned_origin_points = torch.from_numpy(
                np.zeros((bs, 1, 3))).to(src_to_tgt.dtype).to(src_to_tgt.device)
            aligned_origin_points = torch.cat([
                aligned_origin_points[..., :3], torch.ones_like(aligned_origin_points)[..., 0:1]
            ], -1)
            aligned_origin_points = torch.matmul(aligned_origin_points, src_to_tgt)
            batched_origin_points.append(aligned_origin_points[..., :3].contiguous())

        # stack points from different timestamps, and transfer to occupancy representation.
        batched_gt_points = []
        for b in range(bs):
            cur_gt_points = [
                aligned_gt_points[frame_idx][b]
                for frame_idx in range(len(src_frame_idx_list))]
            cur_gt_points = torch.cat(cur_gt_points, 0)
            batched_gt_points.append(cur_gt_points)

        batched_origin_points = torch.cat(batched_origin_points, 1)
        return batched_gt_points, batched_origin_points

    @force_fp32(apply_to=('pred_dict'))
    def loss(self,
             pred_dict,
             gt_points,
             start_idx,
             tgt_bev_h,
             tgt_bev_w,
             tgt_pc_range,
             pred_frame_num,
             img_metas=None,
             suffix='',
             batched_origin_points=None):
        """"Compute loss for all history according to gt_points.

        gt_points: ground-truth point cloud in each frame.
            list of tensor with shape [-1, 5], indicating ground-truth point cloud in
            each frame.
        """
        bev_preds = pred_dict['next_bev_preds']
        valid_frames = np.array(pred_dict['valid_frames'])
        start_frames = (valid_frames + self.history_queue_length + 1 - self.pred_history_frame_num)
        tgt_frames = valid_frames + self.history_queue_length + 1

        full_prev_bev_exists = pred_dict.get('full_prev_bev_exists', True)
        if not full_prev_bev_exists:
            frame_idx_for_loss = [self.pred_history_frame_num] * self.pred_frame_num
        else:
            frame_idx_for_loss = np.arange(0, self.pred_frame_num)

        loss_dict = dict()
        for idx, i in enumerate(frame_idx_for_loss):
            # 1. get the predicted occupancy of frame-i.
            cur_bev_preds = bev_preds[:, :, i, ...].contiguous()

            # 2. get the frame index of current frame.
            src_frames = start_frames + i

            # 3. get gt_points belonging to cur_valid_frames.
            cur_gt_points, cur_origin_points = self._get_reference_gt_points(
                gt_points,
                src_frame_idx_list=src_frames,
                tgt_frame_idx_list=tgt_frames,
                img_metas=img_metas)

            # 4. compute loss.
            if i != self.pred_history_frame_num:
                # For aux history-future supervision:
                #  only compute loss for cur_frame prediction.
                loss_weight = np.array([[1]] + [[0]] * (len(self.loss_weight) - 1))
            else:
                loss_weight = self.loss_weight

            cur_loss_dict = super().loss(
                dict(next_bev_preds=cur_bev_preds,
                     valid_frames=np.arange(0, len(src_frames))),
                cur_gt_points,
                start_idx=start_idx,
                tgt_bev_h=tgt_bev_h,
                tgt_bev_w=tgt_bev_w,
                tgt_pc_range=tgt_pc_range,
                pred_frame_num=len(self.loss_weight)-1,
                img_metas=img_metas,
                batched_origin_points=cur_origin_points,
                loss_weight=loss_weight)

            # 5. merge dict.
            cur_frame_loss_weight = self.per_frame_loss_weight[i]
            cur_frame_loss_weight = cur_frame_loss_weight * (idx == i)
            for k, v in cur_loss_dict.items():
                loss_dict.update({f'{suffix}frame.{idx}.{k}.loss': v * cur_frame_loss_weight})
        return loss_dict

    @force_fp32(apply_to=('pred_dict'))
    def get_point_cloud_prediction(self,
                                   pred_dict,
                                   gt_points,
                                   start_idx,
                                   tgt_bev_h,
                                   tgt_bev_w,
                                   tgt_pc_range,
                                   img_metas=None,
                                   batched_origin_points=None):
        """"Generate point cloud prediction.
        """
        if not self.use_occ_eval:
            # pred_frame_num, inter_num, num_frame, bs, bev_h * bev_w, num_height_pred
            pred_dict['next_bev_preds'] = pred_dict['next_bev_preds'][:, :, self.pred_history_frame_num, ...].contiguous()

        valid_frames = np.array(pred_dict['valid_frames'])
        valid_gt_points, cur_origin_points = self._get_reference_gt_points(
            gt_points,
            src_frame_idx_list=valid_frames + self.history_queue_length + 1,
            tgt_frame_idx_list=valid_frames + self.history_queue_length + 1,
            img_metas=img_metas)

        DEBUG = False
        if DEBUG:
            from .vis_utils import occ_to_voxel, voxel2points
            occ_preds = pred_dict['next_bev_preds']  # (bs, pred_frame_num, 200, 200, 16)

            cur_gt_points = valid_gt_points[0]
            _occ_preds = occ_preds[0]
            for idx in range(_occ_preds.shape[0]):
                occ_pred = _occ_preds[idx]
                print(occ_pred.unique())
                pts, label = voxel2points(occ_pred, ignore_labels=[11])
                np.savetxt(f'results/pts_{idx}_occ.xyz', pts.cpu().numpy())

                chosen_idx = (cur_gt_points[:, -1] == idx)
                curr_pts = cur_gt_points[chosen_idx][:, :3]
                np.savetxt(f'results/pts_{idx}_gt.xyz', curr_pts.cpu().numpy())

        if self.use_occ_eval:
            return super().get_point_cloud_prediction_from_occ(
                pred_dict=pred_dict,
                gt_points=valid_gt_points,
                start_idx=start_idx,
                tgt_bev_h=tgt_bev_h,
                tgt_bev_w=tgt_bev_w,
                tgt_pc_range=tgt_pc_range,
                img_metas=img_metas,
                batched_origin_points=cur_origin_points)
        return super().get_point_cloud_prediction(
            pred_dict=pred_dict,
            gt_points=valid_gt_points,
            start_idx=start_idx,
            tgt_bev_h=tgt_bev_h,
            tgt_bev_w=tgt_bev_w,
            tgt_pc_range=tgt_pc_range,
            img_metas=img_metas,
            batched_origin_points=cur_origin_points)
