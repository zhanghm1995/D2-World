'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-09 15:42:24
Email: haimingzhang@link.cuhk.edu.cn
Description: Add the flow information.
'''
_base_ = [
    '../../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# Dataloader.
future_queue_length_train = 6
future_pred_frame_num_train = 6
rand_frame_interval = (1,)
future_queue_length_test = 6
future_pred_frame_num_test = 6
future_decoder_layer_num = 3
frame_loss_weight = [
    [1],  # for frame 0.
    [1],  # for frame 1.
    [1],  # for frame 2.
    [1],  # for frame 3.
    [1],  # for frame 4.
    [1],  # for frame 5.
    [0]  # ignore.
]
supervise_all_future = True  # set to False for saving GPU memory.
load_frame_interval = 8

# ViDAR model.
vidar_head_pred_history_frame_num = 0
vidar_head_pred_future_frame_num = 0
vidar_head_per_frame_loss_weight = (1.0,)

# latent rendering.
future_latent_render_keep_idx = (),
latent_render_act_fn = 'sigmoid'
latent_render_layer_idx = (2,)
latent_render_grid_step = 0.5

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# nuPlan class samples (not used.)
class_names = ['vehicle', 'bicycle', 'pedestrian',
               'traffic_cone', 'barrier', 'czone_sign', 'generic_object']
num_cams = 8

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 5 # each sequence contains `queue_length` frames.


expansion = 8
model = dict(
    type='ViDARD2WorldWithFlow',
    use_grid_sample=True,
    num_classes=12,
    pred_abs_flow=True,
    history_len=queue_length + 1,
    use_grid_mask=True,
    video_test_mode=True,
    supervise_all_future=supervise_all_future,

    refine_decoder=dict(
        type='SimVP',
        shape_in=(6, 128, 200, 200),
        hid_S=16,
        groups=4,
    ),

    # BEV configuration.
    point_cloud_range=point_cloud_range,
    bev_h=bev_h_,
    bev_w=bev_w_,

    # Predict frame num.
    future_pred_frame_num=future_pred_frame_num_train,
    test_future_frame_num=future_pred_frame_num_test,

    future_pred_head=dict(
        type='D2WorldHead',
        in_channel=expansion*16*2*2,
        d_model=128,
        n_encoder_layers=6,
        n_decoder_layers=6,
        heads=8,
        history_len=queue_length + 1,
        future_len=future_queue_length_train,

        history_queue_length=queue_length,
        pred_history_frame_num=vidar_head_pred_history_frame_num,
        pred_future_frame_num=vidar_head_pred_future_frame_num,
        per_frame_loss_weight=vidar_head_per_frame_loss_weight,

        ray_grid_num=512,
        ray_grid_step=1.0,

        use_ce_loss=True,
        use_dist_loss=False,
        use_dense_loss=True,

        num_pred_fcs=0,  # head for point cloud prediction.
        num_pred_height=16,  # Predict BEV instead of 3D space occupancy.

        can_bus_norm=True,
        can_bus_dims=(0, 1, 2, 17),
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        loss_weight=frame_loss_weight,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        transformer=dict(
            type='PredictionTransformer',
            embed_dims=_dim_,
            decoder=dict(
                type='PredictionDecoder',
                keep_idx=future_latent_render_keep_idx,
                num_layers=future_decoder_layer_num,
                return_intermediate=True,
                transformerlayers=dict(
                    type='PredictionTransformerLayer',
                    attn_cfgs=[
                        # layer-1: deformable self-attention.
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                        # layer-2: deformable cross-attention,
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    latent_render=dict(embed_dims=256, pred_height=16, num_pred_fcs=0,
                                       grid_step=latent_render_grid_step, grid_num=256,
                                       reduction=16, act=latent_render_act_fn, ),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'latent_render', 'ffn', 'norm'))
            )),
    ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'NuPlanD2WorldDataset'
data_split = 'mini'
data_root = f'data/openscene-v1.1/sensor_blobs/{data_split}'
train_ann_pickle_root = f'data/openscene-v1.1/openscene_{data_split}_train_v2.pkl'
val_ann_pickle_root = f'data/openscene-v1.1/openscene_{data_split}_val_v2.pkl'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadNuPlanPointsFromFile',
         coord_type='LIDAR',),
    dict(
        type='LoadNuPlanPointsFromMultiSweeps',
        sweeps_num=0,
        use_dim=[0, 1, 2, 3, 4, 5],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        hard_sweeps_timestamp=0,
        random_select=False,
    ),
    dict(type='LoadOccupancyGT'),
    dict(
        type='CustomVoxelBasedPointSampler',
        cur_sweep_cfg=dict(
            max_num_points=1,
            point_cloud_range=point_cloud_range,
            voxel_size=[1.0, 1.0, 1.0],
            max_voxels=50000,
        ),
        time_dim=4,
    ),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['points', 'aug_param',
                                       'occ_gts'])
]

test_pipeline = [
    dict(type='LoadNuPlanPointsFromFile',
         coord_type='LIDAR',),
    dict(type='LoadOccupancyGT'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['points',
                                       'occ_gts'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_pickle_root,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_img=False,
        use_occ_gts=True,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        future_length=future_queue_length_train,
        rand_frame_interval=rand_frame_interval,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        load_frame_interval=load_frame_interval,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=val_ann_pickle_root,
             use_img=False,
             use_occ_gts=True,
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,

             # some evaluation configuration.
             queue_length=queue_length,
             future_length=future_queue_length_test,
             ego_mask=(-0.8, -1.5, 0.8, 2.5),
             ),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=val_ann_pickle_root,
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality,
              use_img=False,
              use_occ_gts=True,
              # some evaluation configuration.
              queue_length=queue_length,
              future_length=future_queue_length_test,
              ego_mask=(-0.8, -1.5, 0.8, 2.5),
              ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=total_epochs, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=5)
