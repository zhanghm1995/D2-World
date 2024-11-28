'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-05-31 21:50:59
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
import numpy as np
import pickle
import torch
import torch.nn as nn
from . import OPENOCC_LOSS
writer = None

@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs):
        super().__init__()
        
        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)
        
        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0

    def forward(self, inputs):
        
        loss_dict = {}
        tot_loss = 0.
        for loss_func in self.losses:
            loss = loss_func(inputs)
            tot_loss += loss
            loss_dict.update({
                loss_func.__class__.__name__: \
                loss.detach().item() #  / loss_func.weight
            })
            if writer and self.iter_counter % 10 == 0:
                writer.add_scalar(
                    f'loss/{loss_func.__class__.__name__}', 
                    loss.detach().item(), self.iter_counter)
        if writer and self.iter_counter % 10 == 0:
            writer.add_scalar(
                'loss/total', tot_loss.detach().item(), self.iter_counter)
        self.iter_counter += 1
        
        return tot_loss, loss_dict