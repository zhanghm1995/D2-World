'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-22 11:37:19
Email: haimingzhang@link.cuhk.edu.cn
Description: Support only predict the current frame point clouds.
'''
#---------------------------------------------------------------------------------#
# Visual Point Cloud Forecasting enables Scalable Autonomous Driving              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import copy
import torch
import numpy as np

from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from mmcv.parallel import DataContainer as DC

from .nuplan_vidar_dataset_template import NuPlanViDARDatasetTemplate
from .nuplan_vidar_dataset_v1 import NuPlanViDARDatasetV1


@DATASETS.register_module()
class NuPlanViDARDatasetV2(NuPlanViDARDatasetV1):
    r"""Use ViDAR to predict the current point cloud from images frame by frame.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.usable_index = list(range(len(self.data_infos)))

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(self.usable_index[idx])
        while True:

            data = self.prepare_train_data(self.usable_index[idx])
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        ## process the point cloud
        pts_list = [example['points'].data]
        if self.ego_mask is not None:
            pts_list = self._mask_points(pts_list)
        
        # store points in the current frame.
        total_pts_list = []
        for pts in pts_list:
            cur_pts = pts.cpu().numpy().copy()
            cur_pts[:, -1] = self.queue_length
            total_pts_list.append(cur_pts)
        
        example['gt_points'] = DC(
            torch.from_numpy(np.concatenate(total_pts_list, 0)), cpu_only=False)
        example.pop('points')
        return example
    