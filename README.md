# D2-World: An Efficient World Model through Decoupled Dynamic Flow

### [arXiv](https://arxiv.org/abs/2411.17027) | [Talk ](https://opendrivelab.com/cvpr2024/workshop/)  | [Video](https://www.bilibili.com/video/BV19y411v71G/?vd_source=3994b9005446cf917459e6b861cba42b) | [Challenge](https://opendrivelab.com/challenge2024/#predictive_world_model)

## News
- `[2024/11]` Code is released, sorry for so late. 🙏
- `[2024/11]` The technical report is release on [arXiv](https://arxiv.org/abs/2411.17027).
- `[2024/06]` The technical report and talk video are submitted to [Challenge Official Website](https://opendrivelab.com/challenge2024/#predictive_world_model). 
- `[2024/06]` Our method won the Innovation Award💡 and 2nd Place🥈 in the Predictive World Model at [CVPR 2024 Autonomous Grand Challenge](https://opendrivelab.com/challenge2024/). 🎉


## Installation
We tested the code with PyTorch 1.10.0, CUDA 11.3 on Ubuntu 20.04.

Create conda environment with python version 3.8.18
```bash
conda create -n d2world python=3.8.18
conda activate d2world
```
Install the dependencies
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113  -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1 nuscenes-devkit==1.1.10
pip install numba==0.48.0 numpy==1.19.5 pandas==1.3.5 scikit-image==0.19.3 ninja

# Install mmdetection3d from source codes.
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```
Install Detectron2 and Timm.
```bash
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Setup D2World project.
```bash
git clone https://github.com/zhanghm1995/D2-World.git

cd D2-World
mkdir pretrained
cd pretrained & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth

# Install chamferdistance library.
cd third_lib/chamfer_dist/chamferdist/
pip install .
```

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


## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{zhang2024d2world,
  title   = {D$^2$-World: An Efficient World Model through Decoupled Dynamic Flow},
  author  = {Haiming Zhang and Xu Yan and Ying Xue and Zixuan Guo and Shuguang Cui and Zhen Li and Bingbing Liu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2411.17027}
}
```