'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-05 10:53:04
Email: haimingzhang@link.cuhk.edu.cn
Description: Statistics the class frequency of the OpenOCC dataset.
'''
import numpy as np
import os
import os.path as osp
import torch
import mmengine
from tqdm import tqdm


def stat_occ3d_class_freq():
    pkl_fp = "data/nuscenes/bevdetv2-nuscenes_infos_train.pkl"
    data = mmengine.load(pkl_fp)

    data_infos = data['infos']

    occ_dir = "data/nuscenes/gts"
    total_num_cls = 18
    freq_dict = {}
    for i in range(total_num_cls):
        freq_dict[i] = 0
    
    freq_dict[255] = 0

    for info in tqdm(data_infos):
        occ_dir = info['occ_path']
        occupancy_file_path = osp.join(occ_dir, 'labels.npz')
        data = np.load(occupancy_file_path)
        occupancy = torch.tensor(data['semantics'])
        visible_mask = torch.tensor(data['mask_camera'])

        occupancy[~visible_mask.to(torch.bool)] = 255

        # statistics the class frequency
        output, class_freq = occupancy.unique(return_counts=True)

        for idx, cls in enumerate(output.tolist()):
            freq_dict[cls] += class_freq[idx]

    print(freq_dict)


def openscene_occ_to_voxel(occ_data,
                           point_cloud_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0], 
                           occupancy_size=[0.5, 0.5, 0.5],
                           occupancy_classes=11):
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


def stat_openscene_class_freq():
    pkl_fp = "data/openscene-v1.1/openscene_mini_train_v2.pkl"
    data_infos = mmengine.load(pkl_fp)

    total_num_cls = 12
    freq_dict = {}
    for i in range(total_num_cls):
        freq_dict[i] = 0
    
    freq_dict[255] = 0

    for info in tqdm(data_infos):
        occ_gt_path = info['occ_gt_final_path']
        data = np.load(occ_gt_path)
        occupancy = torch.tensor(data)

        occupancy = openscene_occ_to_voxel(occupancy)

        # statistics the class frequency
        output, class_freq = occupancy.unique(return_counts=True)

        for idx, cls in enumerate(output.tolist()):
            freq_dict[cls] += class_freq[idx]

    print(freq_dict)


def stat_openocc_class_freq():
    pkl_fp = "data/nuscenes/bevdetv2-nuscenes_infos_train_openocc.pkl"
    data = mmengine.load(pkl_fp)

    data_infos = data['infos']

    occ_dir = "data/nuscenes/gts"
    total_num_cls = 17
    freq_dict = {}
    for i in range(total_num_cls):
        freq_dict[i] = 0
    
    for info in tqdm(data_infos):
        occupancy_file_path = info['occ_path']
        data = np.load(occupancy_file_path)
        occupancy = torch.tensor(data['semantics'])

        # statistics the class frequency
        output, class_freq = occupancy.unique(return_counts=True)

        for idx, cls in enumerate(output.tolist()):
            freq_dict[cls] += class_freq[idx]

    print(freq_dict)


if __name__ == "__main__":
    stat_openscene_class_freq()

    # stat_occ3d_class_freq()

    # stat_openocc_class_freq()