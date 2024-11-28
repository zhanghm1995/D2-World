'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-18 14:13:28
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import torch
import numpy as np


def voxel2points(voxel, 
                 voxel_size=[0.5, 0.5, 0.5], 
                 range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0], 
                 ignore_labels=[11, 255]):
    if isinstance(voxel, np.ndarray): 
        voxel = torch.from_numpy(voxel)
    
    mask = torch.zeros_like(voxel, dtype=torch.bool)
    for ignore_label in ignore_labels:
        mask = torch.logical_or(voxel == ignore_label, mask)
    mask = torch.logical_not(mask)
    occIdx = torch.where(mask)
    points = torch.cat((occIdx[0][:, None] * voxel_size[0] + voxel_size[0] / 2 + range[0], \
                        occIdx[1][:, None] * voxel_size[1] + voxel_size[1] / 2 + range[1], \
                        occIdx[2][:, None] * voxel_size[2] + voxel_size[2] / 2 + range[2]), dim=1)
    return points, voxel[occIdx]


def occ_to_voxel(occ_data,
                 point_cloud_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0], 
                 occupancy_size=[0.5, 0.5, 0.5],
                 occupancy_classes=11):
    occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
    occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
    occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])

    voxel_num = occ_xdim * occ_ydim * occ_zdim
    
    if occ_data.shape[-1] == 2:
        ## the original occupancy data with shape [N, 2]
        gt_occupancy = np.ones(voxel_num) * occupancy_classes
        gt_occupancy[occ_data[:, 0]] = occ_data[:, 1]

        gt_occupancy = gt_occupancy.reshape(occ_zdim, occ_ydim, occ_xdim)
        gt_occupancy = np.transpose(gt_occupancy, (2, 1, 0))
    else:
        gt_occupancy = occ_data

    print(np.unique(gt_occupancy))
    pc, label = voxel2points(gt_occupancy, ignore_labels=[11])
    return pc, label