'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-15 15:14:22
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import time
import mmengine
import os
from tqdm import tqdm
import os.path as osp
import numpy as np
from collections import defaultdict, OrderedDict
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def stat_data_wo_occ_path():
    """Because the original data has some missing occ_gt_final_path, we need to remove them.
    """
    for split in ['train', 'val']:
        val_pkl_fp = f"data/openscene-v1.1/openscene_mini_{split}.pkl"
        val_meta = mmengine.load(val_pkl_fp)
        print("split:", split, type(val_meta), len(val_meta))

        ## check the occ
        new_val_infos = []
        for info in val_meta:
            occ_gt_path = info['occ_gt_final_path']
            if occ_gt_path is None:
                continue
            new_val_infos.append(info)
        print(len(new_val_infos))



def check_occ():
    val_pkl_fp = "data/openscene-v1.1/openscene_mini_val.pkl"
    val_meta = mmengine.load(val_pkl_fp)
    print(type(val_meta), len(val_meta))

    ## check the occ
    element = val_meta[0]

    occ_gt_path = element['occ_gt_final_path']
    occ_gt = np.load(occ_gt_path)
    print(occ_gt.shape)
    print(np.unique(occ_gt[:, 1]))

    non_empty = occ_gt.shape[0]
    empty_num = 200 * 200 * 16 - non_empty
    print(f"Empty number: {empty_num}, non-empty number: {non_empty}")

    ## check all categories
    all_categories = set()
    for entry in tqdm(val_meta):
        occ_gt_path = entry['occ_gt_final_path']
        if occ_gt_path is None:
            continue

        occ_gt = np.load(occ_gt_path)
        category = np.unique(occ_gt[:, 1])
        all_categories.update(category.tolist())

    print(all_categories)


def check_private_wm_data():
    from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud

    private_test_fp = "data/openscene-v1.1/meta_datas/private_test_wm/private_test_wm.pkl"
    data_infos = mmengine.load(private_test_fp)
    print(type(data_infos), len(data_infos))

    entry = data_infos[0]
    print(entry.keys())

    lidar_path = entry['lidar_path']
    print(lidar_path)

    data_root = "data/openscene-v1.1/sensor_blobs/private_test_wm"
    
    front_cam_path_list = []
    for info in data_infos:
        lidar_path = info['lidar_path']
        occ_gt_path = info['occ_gt_final_path']

        pts_filename = osp.join(data_root, lidar_path)
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T

        front_cam_path = info['cams']['CAM_F0']['data_path']
        front_cam_path_list.append(front_cam_path)

        # for cam_type, cam_info in info['cams'].items():
        #     cam_info['data_path'] = osp.join(data_root, cam_info['data_path'])
        #     print(cam_info['data_path'])
    
    print(front_cam_path_list)


def rewrite_vidar_pred_pc():
    """We resave the vidar predicted point cloud by using the inside masking.
    """
    from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud
    from projects.mmdet3d_plugin.bevformer.utils import e2e_predictor_utils
    
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    pkl_fp = "data/openscene-v1.1/openscene_mini_val_v2.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(type(data_infos), len(data_infos))
    print(data_infos[0].keys())

    our_data_root = "results/vidar_pred_pc"
    data_root = "data/openscene-v1.1/sensor_blobs/mini"

    save_root = "results/vidar_pred_pc_new"
    for idx, info in tqdm(enumerate(data_infos)):
        lidar_path = info['lidar_path']
        pts_filename = osp.join(data_root, lidar_path)
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T

        our_pc = osp.join(our_data_root, lidar_path + ".npz")
        our_pc = np.load(our_pc)['arr_0']

        gt_inside_mask = e2e_predictor_utils.get_inside_mask(our_pc, point_cloud_range)
        our_pc = our_pc[gt_inside_mask]
        
        save_path = os.path.join(save_root, lidar_path)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(save_path, our_pc)


def rewrite_occupancy():
    from projects.mmdet3d_plugin.bevformer.utils.occ_utils import occ_to_voxel

    pkl_fp = "data/openscene-v1.1/openscene_trainval_val_v2_valid.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(type(data_infos), len(data_infos))

    ## including the previous frames
    save_root = "results/xworld_occ_w_flow_trainval_val_e20_all_origin"
    for data in tqdm(data_infos):
        occ_gt_path = data['occ_gt_final_path']
        save_path = occ_gt_path.replace('dataset/openscene-v1.0', save_root)
        save_path = save_path.replace('data/openscene-v1.0', save_root)
        save_dir = osp.split(save_path)[0]
        os.makedirs(save_dir, exist_ok=True)

        occ_data = np.load(occ_gt_path)
        occ = occ_to_voxel(occ_data).numpy()
        np.savez_compressed(save_path, occ.astype(np.uint8))



def check_vidar_pred_pc():
    """Save the groud truth point cloud and the predicted point cloud for comparison.
    """
    from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud
    from projects.mmdet3d_plugin.bevformer.utils import e2e_predictor_utils
    
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    pkl_fp = "data/openscene-v1.1/openscene_mini_train.pkl"
    pkl_fp = "data/openscene-v1.1/openscene_mini_train_v2.pkl"
    # pkl_fp = "data/openscene-v1.1/openscene_mini_val_v2.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(type(data_infos), len(data_infos))
    print(data_infos[0].keys())

    our_data_root = "results/vidar_pred_pc"
    our_data_root = "/data1/cyx/OpenScene/results/vidar_pred_pc_train"
    data_root = "data/openscene-v1.1/sensor_blobs/mini"
    for idx, info in enumerate(data_infos):
        lidar_path = info['lidar_path']
        pts_filename = osp.join(data_root, lidar_path)
        pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T

        our_pc = osp.join(our_data_root, lidar_path + ".npz")
        our_pc = np.load(our_pc)['arr_0']

        gt_inside_mask = e2e_predictor_utils.get_inside_mask(our_pc, point_cloud_range)
        our_pc = our_pc[gt_inside_mask]

        save_dir = "results/vidar_pred_compare_train"
        os.makedirs(save_dir, exist_ok=True)
        
        np.savetxt(f"{save_dir}/{idx:03d}_pc_gt.xyz", pc[:, :3])
        np.savetxt(f"{save_dir}/{idx:03d}_pc_ours.xyz", our_pc[:, :3])
        
        if idx > 10:
            break


def load_lidar_pc():
    from projects.mmdet3d_plugin.datasets.pipelines.nuplan_loading import PointCloud
    from projects.mmdet3d_plugin.bevformer.utils import e2e_predictor_utils

    min_lidar_fp = "data/openscene-v1.1/sensor_blobs/mini/2021.05.12.22.28.35_veh-35_00620_01164/MergedPointCloud/00a0fec4c02f5f05.pcd"
    train_lidar_fp = "data/openscene-v1.1/sensor_blobs/trainval/2021.05.12.22.28.35_veh-35_00620_01164/MergedPointCloud/00a0fec4c02f5f05.pcd"
    
    iters = 200
    
    total_time = 0.0
    for i in tqdm(range(iters)):
        start = time.time()
        pc = PointCloud.parse_from_file(min_lidar_fp).to_pcd_bin2().T
        end = time.time()
        elapsed = end - start
        total_time += elapsed
    print("Mini Average time taken: ", total_time / iters)

    total_time = 0.0
    for i in tqdm(range(iters)):
        start = time.time()
        pc = PointCloud.parse_from_file(train_lidar_fp).to_pcd_bin2().T
        end = time.time()
        elapsed = end - start
        total_time += elapsed
    print("Train Average time taken: ", total_time / iters)

    total_time = 0.0
    for i in tqdm(range(iters)):
        start = time.time()
        pc = np.load("pcccc.npz")['arr_0']
        end = time.time()
        elapsed = end - start
        total_time += elapsed
    print("NPZ Average time taken: ", total_time / iters)

    return

    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    data_part = 'trainval'
    pkl_fp = f"data/openscene-v1.1/openscene_{data_part}_train.pkl"
    pkl_fp = "data/openscene-v1.1/openscene_mini_train_v2.pkl"
    # pkl_fp = "data/openscene-v1.1/openscene_mini_val_v2.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(type(data_infos), len(data_infos))
    print(data_infos[0].keys())

    data_root = f"data/openscene-v1.1/sensor_blobs/{data_part}"

    info = data_infos[10]
    lidar_path = info['lidar_path']
    pts_filename = osp.join(data_root, lidar_path)
    
    start = time.time()
    pc = PointCloud.parse_from_file(pts_filename).to_pcd_bin2().T
    print(f"Time cost: {time.time() - start:.7f}s")
    print(pc.shape)


def check_data_length():
    data_part = 'trainval'
    pkl_fp = "data/openscene-v1.1/openscene_mini_train.pkl"
    # pkl_fp = f"data/openscene-v1.1/openscene_{data_part}_train_v2.pkl"
    pkl_fp = "data/openscene-v1.1/openscene_mini_train_v2.pkl"

    data_infos = mmengine.load(pkl_fp)
    print(type(data_infos), len(data_infos))
    print(data_infos[0].keys())

    # convert the list of dict to dict
    data_infos_dict = defaultdict(list)
    for d in data_infos:
        data_infos_dict[d["scene_name"]].append(d)

    data_infos_dict = dict(data_infos_dict)
    print(len(data_infos_dict))

    num_frames_each_scene = [len(_scene) for _scene in data_infos_dict.values()]
    print(min(num_frames_each_scene), max(num_frames_each_scene))
    print(sorted(num_frames_each_scene)[:100])

    filtered_scene_names = []
    for key, value in data_infos_dict.items():
        # filter the scenes with less than 12 frames
        if len(value) < 12:
            continue
        filtered_scene_names.append(key)

    # only keep the first 1/4 for acceleration
    filtered_data_infos = []
    for key in filtered_scene_names:
        filtered_data_infos.extend(data_infos_dict[key])
    print(f"Filtered data length: {len(filtered_data_infos)}")

    pkl_file_path = f"data/openscene-v1.1/openscene_{data_part}_train_v2_filter.pkl"
    with open(pkl_file_path, "wb") as f:
        pickle.dump(filtered_data_infos, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_scene_list(data_infos, scene_flag="scene_name"):
    # convert the list of dict to dict
    data_infos_dict = defaultdict(list)
    for d in data_infos:
        data_infos_dict[d[scene_flag]].append(d)

    data_infos_dict = dict(data_infos_dict)
    print(f"scene numbers: {len(data_infos_dict)}")

    return data_infos_dict


def save_partial_data(data_part, split):
    """Save partial dataset for fast validation.

    Args:
        data_part (str): trainval or mini
        split (str): train or val or test
    """
    pkl_fp = f"data/openscene-v1.1/openscene_{data_part}_{split}_v2.pkl"
    print(pkl_fp)

    meta = mmengine.load(pkl_fp)
    print(type(meta), len(meta))

    scene_infos_list = get_scene_list(meta, scene_flag='scene_token')
    print(len(scene_infos_list))
    for scene_info, data in scene_infos_list.items():
        print(scene_info, len(data))

    ## only keep the scene with more than 10 data
    keep_length = 5000
    kept_data_infos = []
    for scene_info, data in scene_infos_list.items():
        if len(data) > 10:
            kept_data_infos.extend(data)
        
        if len(kept_data_infos) > keep_length:
            break
    
    print(len(kept_data_infos))
    save_pkl_fp = f"data/openscene-v1.1/openscene_{data_part}_{split}_valid.pkl"
    with open(save_pkl_fp, "wb") as f:
        pickle.dump(kept_data_infos, f, protocol=pickle.HIGHEST_PROTOCOL)


def check_pickle_length():
    pkl_fp = "data/openscene-v1.1/openscene_mini_val.pkl"
    data = mmengine.load(pkl_fp)
    print(type(data), len(data))


if __name__ == "__main__":
    check_pickle_length()
    exit()
    # load_lidar_pc()
    # save_partial_data('trainval', 'val')
    rewrite_occupancy()
    # check_vidar_pred_pc()
    # rewrite_vidar_pred_pc()
    exit()
    

    
    # check_private_wm_data()



# private_test_fp = "data/openscene-v1.1/meta_datas/private_test_wm/private_test_wm.pkl"
# private_test_fp = "data/openscene-v1.1/openscene_mini_train.pkl"

# private_test_meta = mmengine.load(private_test_fp)
# print(type(private_test_meta), len(private_test_meta))
# print(private_test_meta[0].keys())
# exit()


# private_test_fp = "data/openscene-v1.1/meta_datas/private_test_wm/private_test_wm.pkl"
# private_test_meta = mmengine.load(private_test_fp)
# print(type(private_test_meta), len(private_test_meta))
# print(private_test_meta[0].keys())

# _meta = private_test_meta[0]
# print(_meta['frame_idx'])

# if 'pts_filename' in private_test_meta[0]:
#     print(private_test_meta[0]['pts_filename'])
# else:
#     print('No pts_filename in the meta file.')