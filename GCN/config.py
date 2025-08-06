# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config(exp_id):

    cfg = Config()

    """实验配置"""
    cfg.experiment_idx = exp_id
    cfg.trial_id = None

    cfg.save_dir_prefix = 'Experiment_' # prefix for experiment folder
    cfg.name = 'voxel2mesh'

    """数据集配置"""
    cfg.patch_shape = (64, 64, 64)

    cfg.ndims = 2
    cfg.augmentation_shift_range = 10

    """模型配置"""
    cfg.first_layer_channels = 32
    cfg.num_input_channels = 3
    cfg.steps = 4

    # Only supports batch size 1 at the moment.
    cfg.batch_size = 1

    cfg.num_classes = 19
    cfg.batch_norm = True
    cfg.graph_conv_layer_count = 4

    """优化器"""
    cfg.learning_rate = 1e-4

    """训练配置"""
    cfg.numb_of_itrs = 300000
    cfg.eval_every = 1000 # saves results to disk

    return cfg
