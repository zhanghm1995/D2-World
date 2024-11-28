'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-11 10:15:25
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import numpy as np
import torch


def occ_to_voxel(occ_data,
                 point_cloud_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0], 
                 occupancy_size=[0.5, 0.5, 0.5],
                 occupancy_classes=11):
    """Convert the occupancy label from OpenScene dataset into voxel format.

    Args:
        occ_data (Tensor | np.ndarray): (N, 2), representing the voxel indices and the corresponding class labels.
        point_cloud_range (list, optional): _description_. Defaults to [-50.0, -50.0, -4.0, 50.0, 50.0, 4.0].
        occupancy_size (list, optional): _description_. Defaults to [0.5, 0.5, 0.5].
        occupancy_classes (int, optional): _description_. Defaults to 11.

    Returns:
        Tensor: the voxelized occupancy with shape [X, Y, Z]
    """
    if isinstance(occ_data, np.ndarray): 
        occ_data = torch.from_numpy(occ_data)

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