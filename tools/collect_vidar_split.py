'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-16 19:34:13
Email: haimingzhang@link.cuhk.edu.cn
Description: Collect the mini split used by ViDAR. https://github.com/OpenDriveLab/ViDAR/issues/20
'''

import os, sys
import pickle


def save_infos(files, split):
    infos = []
    for file in files:
        with open(file, 'rb') as f:
            infos.extend(pickle.load(f))

    with open(f'data/openscene-v1.1/openscene_{split}.pkl', 'wb') as f:
        pickle.dump(infos, f, protocol=pickle.HIGHEST_PROTOCOL)


def collect_trainval_split():
    split = 'trainval'
    paths = os.listdir(f'data/openscene-v1.1/meta_datas/{split}')
    paths = [
        os.path.join(f'data/openscene-v1.1/meta_datas/{split}', each)
        for each in paths if each.endswith('.pkl')]
    print(f'{split}:', len(paths))
          
    if split == 'test':
        save_infos(paths, 'test')
    else:
        train_paths = paths[:int(len(paths) * 0.85)]
        print(f"{split}_train: {len(train_paths)}")
        save_infos(train_paths, f'{split}_train')

        val_paths = paths[int(len(paths) * 0.85):]
        print(f"{split}_val: {len(val_paths)}")
        save_infos(val_paths, f'{split}_val')
    

def collect_mini_split():
    split = 'mini'
    
    # train split
    train_split_file = f'data/{split}_train.txt'

    with open(train_split_file, 'r') as f:
        train_paths = [line.rstrip() for line in f.readlines()]

    train_paths = [
        os.path.join(f'data/openscene-v1.1/meta_datas/{split}', f'{each}.pkl')
        for each in train_paths]
    print('train:', len(train_paths))
    save_infos(train_paths, f'{split}_train')
    
    # validation split
    val_split_file = f'data/{split}_val.txt'
    with open(val_split_file, 'r') as f:
        val_paths = [line.rstrip() for line in f.readlines()]
    
    val_paths = [
        os.path.join(f'data/openscene-v1.1/meta_datas/{split}', f'{each}.pkl')
        for each in val_paths]
    print('val:', len(val_paths))
    save_infos(val_paths, f'{split}_val')


if __name__ == "__main__":
    split = sys.argv[1]

    if split == 'mini':
        collect_mini_split()
    elif split == 'trainval':
        collect_trainval_split()
    else:
        raise ValueError(f"Invalid split: {split}")

    print('Done!')