'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-17 09:48:46
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import mmengine
import os
import os.path as osp
import numpy as np
import torch
import os, sys
from tqdm import tqdm
import pickle


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def occupancy2pointcloud(occupancy,
                         free_cls_idx,
                         pc_range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0],
                         voxel_size=[0.5, 0.5, 0.5]):
    occupancy[occupancy == free_cls_idx] = 0
    fov_voxels = np.stack(occupancy.nonzero())  # (3, N)
    fov_voxels = fov_voxels.transpose((1, 0))  # to (N, 3)

    fov_voxels = fov_voxels.astype(np.float32)

    fov_voxels[:, :3] = (fov_voxels[:, :3].astype(np.float32) + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]
    return fov_voxels


def voxel2points(voxel, 
                 voxel_size=[0.5, 0.5, 0.5], 
                 range=[-50.0, -50.0, -4.0, 50.0, 50.0, 4.0], 
                 ignore_labels=[11]):
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

    gt_occupancy = np.ones(voxel_num) * occupancy_classes
    gt_occupancy[occ_data[:, 0]] = occ_data[:, 1]
    print(gt_occupancy.shape)

    gt_occupancy = gt_occupancy.reshape(occ_zdim, occ_ydim, occ_xdim)
    print(gt_occupancy.shape, np.unique(gt_occupancy))

    gt_occupancy = np.transpose(gt_occupancy, (2, 1, 0))
    print(gt_occupancy.shape)

    pc, label = voxel2points(gt_occupancy, ignore_labels=[11])
    print(pc.shape, label.shape)
    np.savetxt("data/occ_pc2.xyz", pc)



def add_occ_path():
    from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud

    pkl_fp = "data/openscene-v1.1/openscene_mini_train.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(len(data_infos))
    
    element = data_infos[0]
    print(element.keys())
    occ_gt_final_path = element['occ_gt_final_path']
    print(occ_gt_final_path)

    occ = np.load(occ_gt_final_path)
    print(occ.shape, np.unique(occ[:, 1]))

    # occ_to_voxel(occ)
    
    print(element['lidar2ego_translation'])
    print(element['lidar2ego_rotation'])
    # load lidar point cloud
    data_split = 'mini'
    data_root = f'data/openscene-v1.1/sensor_blobs/{data_split}'
    pts_filename=os.path.join(data_root, element['lidar_path'])
    pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T
    print(pc.shape)
    print(np.unique(pc[:, 5]))
    np.savetxt("data/pc_origin.xyz", pc[:, :3])


    # gt_occupancy = np.ones((1*seq_len, self.voxel_num), dtype=torch.long)*self.occupancy_classes)
    # for sample_id in range(len(temporal_gt_bboxes_list)):
    #     for frame_id in range(seq_len):
    #         occ_gt = occ_gts[sample_id][frame_id].long()
    #         gt_occupancy[sample_id*seq_len+frame_id][[occ_gt[:, 0]]] = occ_gt[:, 1]


def remove_data_wo_occ_path():
    """Because the original data has some missing occ_gt_final_path, we need to remove them.
    """
    # data_part = 'mini'
    data_part = "trainval"

    for split in ['train', 'val']:
        val_pkl_fp = f"data/openscene-v1.1/openscene_{data_part}_{split}.pkl"
        meta = mmengine.load(val_pkl_fp)
        print("split:", split, type(meta), len(meta))

        ## check the occ
        new_val_infos = []
        for info in tqdm(meta):
            occ_gt_path = info['occ_gt_final_path']
            if occ_gt_path is None:
                continue
            new_val_infos.append(info)
        print(len(new_val_infos))

        mmengine.dump(new_val_infos, 
                      f"data/openscene-v1.1/openscene_{data_part}_{split}_v2.pkl",
                      protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    remove_data_wo_occ_path()
    exit()
    add_occ_path()