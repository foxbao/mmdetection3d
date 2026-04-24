import sys
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


from pytorch_quantization.nn.modules.quant_conv import QuantConv2d, QuantConvTranspose2d

from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn

import lean.quantize as quantize
import lean.funcs as funcs

from torch.utils.data import DataLoader


import warnings
from copy import deepcopy
from os import path as osp
from pathlib import Path
from typing import Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmdet3d.structures.det3d_data_sample import SampleList

from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)

from mmengine.runner import Runner

def fuse_conv_bn(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, QuantConv2d) or isinstance(child, nn.Conv2d): # or isinstance(child, QuantConvTranspose2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module

def quantize_net(model):
    quantize.quantize_encoders_lidar_branch(model.pts_middle_encoder)    
    # quantize.quantize_encoders_camera_branch(model.encoders.camera)
    # quantize.replace_to_quantization_module(model.fuser)
    quantize.quantize_decoder(model.pts_backbone)
    quantize.quantize_decoder(model.pts_neck)
    model.pts_middle_encoder = funcs.layer_fusion_bn(model.pts_middle_encoder)
    return model



def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Quantization Aware Training")
    parser.add_argument('--config', type=str, help='Config file')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference')
    parser.add_argument('--save_path', type=str, default='projects/BEVFusion/qat/ckpt/ptq_bevfusion.pth', help='Path to save the quantized model')
    return parser.parse_args()


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               palette: str = 'none',
               cfg_options: Optional[dict] = None):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    convert_SyncBN(config.model)
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmdet3d'))
    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmdet3d 1.x
            model.dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmdet3d 1.x
            classes = checkpoint['meta']['CLASSES']
            model.dataset_meta = {'classes': classes}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
        else:
            # < mmdet3d 1.x
            model.dataset_meta = {'classes': config.class_names}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']

        test_dataset_cfg = deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        warnings.warn('Don\'t suggest using CPU device. '
                      'Some functions are not supported for now.')

    model.to(device)
    model.eval()
    dataloader_cfg = copy.deepcopy(config.test_dataloader)
    data_loader = Runner.build_dataloader(dataloader_cfg)

    return model, data_loader


def main(args):
    quantize.initialize()
    model, data_loader = init_model(args.config, args.checkpoint, device=args.device)
    model = quantize_net(model)
    model = fuse_conv_bn(model)
    print("🔥 start calibrate 🔥 ")
    quantize.set_quantizer_fast(model)
    quantize.calibrate_model(model, data_loader, 0, None, 5)

    quantize.disable_quantization(model.pts_middle_encoder.conv_input).apply()
    quantize.disable_quantization(model.pts_neck.deblocks[0][0]).apply()
    quantize.print_quantizer_status(model)
    print(f"Done due to ptq only! Save checkpoint to {args.save_path} 🤗")
    model.pts_middle_encoder = funcs.fuse_relu_only(model.pts_middle_encoder)
    torch.save(model, args.save_path)
    return



if __name__ == "__main__":
    args = parse_args()
    main(args)