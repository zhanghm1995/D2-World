# D2-World: An Efficient World Model through Decoupled Dynamic Flow

### [arXiv](https://arxiv.org/abs/2411.17027) | [Talk ](https://opendrivelab.com/cvpr2024/workshop/)  | [Video](https://www.bilibili.com/video/BV19y411v71G/?vd_source=3994b9005446cf917459e6b861cba42b) | [Challenge](https://opendrivelab.com/challenge2024/#predictive_world_model)

## News
- `[2024/11]` Code is released, sorry for so late. 🙏
- `[2024/11]` The technical report is release on [arXiv](https://arxiv.org/abs/2411.17027).
- `[2024/06]` The technical report and talk video are submitted to [Challenge Official Website](https://opendrivelab.com/challenge2024/#predictive_world_model). 
- `[2024/06]` Our method won the Innovation Award💡 and 2nd Place🥈 in the Predictive World Model at [CVPR 2024 Autonomous Grand Challenge](https://opendrivelab.com/challenge2024/). 🎉


## Prepare Dataset
### 1. Download Dataset
Because the whole OpenScene-v1.1 dataset is very large, we can use the mini-split version of the dataset for training and evaluation at the beginning. 

You can download the OpenScene-v1.1 dataset from [OpenXLab](https://openxlab.org.cn/datasets/OpenDriveLab/OpenScene/cli/main). We recommend using the CLI tool to download the dataset. Please refer the instructions on the website and then excute the following commands to download the dataset:
```bash
# camera data
openxlab dataset download --dataset-repo OpenDriveLab/OpenScene --source-path /openscene-v1.1/openscene_sensor_mini_camera.tgz --target-path ./

# lidar data
openxlab dataset download --dataset-repo OpenDriveLab/OpenScene --source-path /openscene-v1.1/openscene_sensor_mini_lidar.tgz --target-path ./
```
And the metadata from Google Drive: [Metadata](https://drive.google.com/drive/folders/1MnRwhnEBsgZxbaleHxc3Gw7Ovc4I9az1).

In D2-World, we use the occupancy as the intermediate representation, the occupancy data can be obtained from vision-centric methods like BEVDet during inference stage. Also, we could use the occupancy groudtruth data from OpenScene-v1.0 dataset to train our model for convenience. 

You can download the occupancy labels from the OpenScene-v1.0 dataset:
```bash
openxlab dataset download --dataset-repo OpenDriveLab/OpenScene --source-path /openscene-v1.0/occupancy/mini/occ_mini.tar.gz --target-path ./
```

After downloading the dataset, you can create the symbolic links to the dataset folder:
```bash
cd data
ln -s /path/to/openscene-v1.1 ./


# for the occupancy labels
cd D2World
mkdir dataset
ln -s /path/to/openscene-v1.0 ./
```

### 2. Preprocess the Dataset
- Collect the dataset split
```bash
python tools/collect_vidar_split.py mini
```
- Remove some invalid samples without the occupancy labels
```bash
python tools/update_pickle.py
```

## Train the Model
```bash
bash tools/dist_train.sh projects/configs/vidar_pretrain/OpenScene/xworld_OpenScene_mini_1_8_3future_binary_new.py 8
```

## Evaluate the Model
```bash
bash tools/dist_test.sh <path/to/config.py> work_dirs/epoch_xxx.pth 8
```


## Acknowledgements
Many thanks to the following great open-source repositories:
+ [ViDAR](https://github.com/OpenDriveLab/ViDAR)
+ [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
+ [OccWorld](https://github.com/wzzheng/OccWorld)