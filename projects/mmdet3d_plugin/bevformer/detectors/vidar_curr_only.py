'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-22 12:58:54
Email: haimingzhang@link.cuhk.edu.cn
Description: Only predict the current point cloud from images.
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
from .vidar import ViDAR


def merge_close_vectors_kd_tree(vectors, threshold=0.1):
  """使用 KD 树合并相近的单位方向向量。

  Args:
    vectors: Nx3 的单位方向向量数组。
    threshold: 向量夹角的余弦值阈值，小于该阈值则认为两个向量相近。

  Returns:
    Mx3 的合并后的单位方向向量数组。
  """
  from sklearn.neighbors import KDTree

  # 创建 KD 树
  tree = KDTree(vectors)

  # 初始化合并后的向量列表
  merged_vectors = []
  merged = set()

  # 遍历所有向量
  for i in range(len(vectors)):
    # 如果当前向量已被合并，则跳过
    if i in merged:
      continue

    # 使用 KD 树搜索与当前向量夹角余弦值大于阈值的近邻向量
    close_vectors = tree.query_radius(vectors[i].reshape(1, -1), r=np.arccos(threshold), return_distance=False)[0]

    # 将当前向量和所有与之相近的向量合并
    merged_vectors.append(np.mean(vectors[close_vectors], axis=0))

    # 将这些向量标记为已合并
    merged.update(close_vectors)

  # 将合并后的向量归一化
  merged_vectors = np.array(merged_vectors)
  merged_vectors = merged_vectors / np.linalg.norm(merged_vectors, axis=1, keepdims=True)

  return merged_vectors

@DETECTORS.register_module()
class ViDARCurrOnly(ViDAR):
    def __init__(self,
                 save_offline=True,
                 save_root="/data1/cyx/OpenScene/results/vidar_pred_pc_val_debug",
                 *args,
                 **kwargs,):

        super().__init__(*args, **kwargs)
        self.save_offline = save_offline
        self.save_root = save_root

    def forward_test(self, 
                     img_metas, 
                     img=None,
                     gt_points=None, 
                     **kwargs):
        """has similar implementation with train forward."""
        # 1. Extract history BEV features.
        img = [img]
        img_metas = [img_metas]

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        
        prev_bev = new_prev_bev

        # 2. Align previous frames to reference coordinates.
        prev_bev = prev_bev[:, None, ...].contiguous()  # bs, 1, bev_h * bev_w, c
        next_bev_feats, valid_frames = [], []
        ref_bev = prev_bev[:, -1].contiguous()
        next_bev_feats.append(ref_bev.unsqueeze(0).repeat(
            len(self.future_pred_head.bev_pred_head), 1, 1, 1
        ).contiguous())
        valid_frames.append(0)

        # 3. predict future BEV.
        valid_frames.extend(list(range(1, self.test_future_frame_num+1)))
        # img_metas = [each[num_frames - 1] for each in img_metas]

        # pred_frame_num, inter_num, bs, bev_h * bev_w, embed_dims.
        next_bev_feats = torch.stack(next_bev_feats, 0)
        next_bev_preds = self.future_pred_head.forward_head(next_bev_feats)
        pred_dict = {
            'next_bev_features': next_bev_feats,
            'next_bev_preds': next_bev_preds,
            'valid_frames': valid_frames,
        }

        img_metas = img_metas[0]
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

        if self.save_offline:
            ## save the predicted point cloud
            origin_pts_fp = img_metas[0]['pts_filename']
            data_root = f"data/openscene-v1.1/sensor_blobs/mini"
            pts_filename = origin_pts_fp.replace(data_root, "")
            pts_filename = pts_filename[1:] if pts_filename.startswith("/") else pts_filename

            save_path = os.path.join(self.save_root, pts_filename)
            save_dir = os.path.dirname(save_path)
            mmcv.mkdir_or_exist(save_dir)

            gt_inside_mask = e2e_predictor_utils.get_inside_mask(pred_pcds[0][0], self.point_cloud_range)
            pred_pcd_inside = pred_pcds[0][0][gt_inside_mask]
            
            pred_pcd = pred_pcd_inside.cpu().numpy()
            np.savez_compressed(save_path, pred_pcd)
            # np.savetxt(f"{img_metas[0]['sample_idx']}_pred.xyz", pred_pcd)
            return [None]

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

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        outs = self.pts_bbox_head(
            img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return outs
    
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

    
