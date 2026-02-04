
import argparse
import copy
import os
import warnings
from easydict import EasyDict
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union
from collections import abc
import functools
from inspect import getfullargspec

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnxsim import simplify

# custom functional package
import lean.funcs as funcs
import lean.exptool as exptool
import lean.quantize as quantize


from pytorch_quantization.nn.modules.quant_conv import QuantConv2d, QuantConvTranspose2d
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer


from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn
from mmcv.cnn import fuse_conv_bn


import mmengine
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint


from mmengine.registry import (DATASETS,
                               MODELS)
from mmengine.runner import Runner
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version

from mmdet3d.registry import DATASETS, MODELS

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.autocast would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    # Note that when PyTorch >= 1.6.0, we still cast tensor types to fp16
    # manually, so the behavior may not be consistent with real amp.
    from torch.cuda.amp import autocast
except ImportError:
    pass

def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored. If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp16
        >>>     @auto_fp16()
        >>>     def forward(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp16
        >>>     @auto_fp16(apply_to=('pred', ))
        >>>     def do_something(self, pred, others):
        >>>         pass
    """

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=True):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper



class SubclassFuser(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @auto_fp16(apply_to=("features",))
    def forward(self, features):
        if self.parent.fusion_layer is not None:
            x = self.parent.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.parent.pts_backbone(x)
        x = self.parent.pts_neck(x)
        return x[0]


class SubclassHead(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.classes_eye  = nn.Parameter(torch.eye(parent.bbox_head.num_classes).float())
    
    @staticmethod
    @auto_fp16(apply_to=("inputs", "classes_eye"))
    def head_forward(self, inputs, classes_eye):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = int(inputs.shape[0])
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, int(lidar_feat.shape[1]), -1
        )  # [BS, C, H*W]
        bev_pos = self.bev_pos.to(lidar_feat.dtype).repeat(batch_size, 1, 1).to(lidar_feat.device)

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            # local_max[
            #     :,
            #     8,
            # ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     9,
            # ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            local_max[:, 8] = heatmap[:, 8]
            local_max[:, 9] = heatmap[:, 9]
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            # local_max[
            #     :,
            #     1,
            # ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     2,
            # ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            local_max[:, 1] = heatmap[:, 1]
            local_max[:, 2] = heatmap[:, 2]
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, int(heatmap.shape[1]), -1)

        # top #num_proposals among all classes
        # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        #     ..., : self.num_proposals
        # ]
        top_proposals = heatmap.view(batch_size, -1).topk(k=self.num_proposals, dim=-1, largest=True)[1]
        top_proposals_class = top_proposals // int(heatmap.shape[-1])
        top_proposals_index = top_proposals % int(heatmap.shape[-1])
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        # self.one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
        #     0, 2, 1
        # ).half()
        self.one_hot = classes_eye.index_select(0, top_proposals_class.view(-1))[None].permute(
            0, 2, 1
        )
        query_cat_encoding = self.class_encoding(self.one_hot)
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat, lidar_feat_flatten, query_pos, bev_pos
            )

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return ret_dicts[-1]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                )
            else:
                new_res[key] = ret_dicts[0][key]
        return new_res

    def get_bboxes(self, preds_dict, one_hot):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        # batch_score = preds_dict["heatmap"][..., -self.num_proposals :].sigmoid()
        batch_score = preds_dict["heatmap"].sigmoid()
        # if self.loss_iou.loss_weight != 0:
        #    batch_score = torch.sqrt(batch_score * preds_dict['iou'][..., -self.num_proposals:].sigmoid())
        # one_hot = F.one_hot(
        #     query_labels, num_classes=num_classes
        # ).permute(0, 2, 1)
        batch_score = batch_score * preds_dict["query_heatmap_score"] * one_hot
        # batch_center = preds_dict["center"][..., -self.num_proposals :]
        # batch_height = preds_dict["height"][..., -self.num_proposals :]
        # batch_dim = preds_dict["dim"][..., -self.num_proposals :]
        # batch_rot = preds_dict["rot"][..., -self.num_proposals :]
        # batch_vel = None
        # if "vel" in preds_dict:
        #     batch_vel = preds_dict["vel"][..., -self.num_proposals :]
        batch_center = preds_dict["center"]
        batch_height = preds_dict["height"]
        batch_dim = preds_dict["dim"]
        batch_rot = preds_dict["rot"]
        batch_vel = None
        if "vel" in preds_dict:
            batch_vel = preds_dict["vel"]

        return [batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel]

    def forward(self, x):
        for type, head in self.parent.heads.items():
            if type == "object":
                pred_dict = self.head_forward(head, x, self.classes_eye)
                return self.get_bboxes(pred_dict, head.one_hot)
            else:
                raise ValueError(f"unsupported head: {type}")
            
    @staticmethod          
    @auto_fp16(apply_to=("inputs", "classes_eye"))
    def head_forward(self, inputs, classes_eye):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = int(inputs.shape[0])
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, int(lidar_feat.shape[1]), -1
        )  # [BS, C, H*W]
        bev_pos = self.bev_pos.to(lidar_feat.dtype).repeat(batch_size, 1, 1).to(lidar_feat.device)

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            # local_max[
            #     :,
            #     8,
            # ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     9,
            # ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            local_max[:, 8] = heatmap[:, 8]
            local_max[:, 9] = heatmap[:, 9]
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            # local_max[
            #     :,
            #     1,
            # ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     2,
            # ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            local_max[:, 1] = heatmap[:, 1]
            local_max[:, 2] = heatmap[:, 2]
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, int(heatmap.shape[1]), -1)

        # top #num_proposals among all classes
        # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        #     ..., : self.num_proposals
        # ]
        top_proposals = heatmap.view(batch_size, -1).topk(k=self.num_proposals, dim=-1, largest=True)[1]
        top_proposals_class = top_proposals // int(heatmap.shape[-1])
        top_proposals_index = top_proposals % int(heatmap.shape[-1])
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        # self.one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
        #     0, 2, 1
        # ).half()
        self.one_hot = classes_eye.index_select(0, top_proposals_class.view(-1))[None].permute(
            0, 2, 1
        )
        query_cat_encoding = self.class_encoding(self.one_hot)
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat, lidar_feat_flatten, query_pos=query_pos, key_pos=bev_pos
            )

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return ret_dicts[-1]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                )
            else:
                new_res[key] = ret_dicts[0][key]
        return new_res

    def get_bboxes(self, preds_dict, one_hot):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        # batch_score = preds_dict["heatmap"][..., -self.num_proposals :].sigmoid()
        batch_score = preds_dict["heatmap"].sigmoid()
        # if self.loss_iou.loss_weight != 0:
        #    batch_score = torch.sqrt(batch_score * preds_dict['iou'][..., -self.num_proposals:].sigmoid())
        # one_hot = F.one_hot(
        #     query_labels, num_classes=num_classes
        # ).permute(0, 2, 1)
        batch_score = batch_score * preds_dict["query_heatmap_score"] * one_hot
        # batch_center = preds_dict["center"][..., -self.num_proposals :]
        # batch_height = preds_dict["height"][..., -self.num_proposals :]
        # batch_dim = preds_dict["dim"][..., -self.num_proposals :]
        # batch_rot = preds_dict["rot"][..., -self.num_proposals :]
        # batch_vel = None
        # if "vel" in preds_dict:
        #     batch_vel = preds_dict["vel"][..., -self.num_proposals :]
        batch_center = preds_dict["center"]
        batch_height = preds_dict["height"]
        batch_dim = preds_dict["dim"]
        batch_rot = preds_dict["rot"]
        batch_vel = None
        if "vel" in preds_dict:
            batch_vel = preds_dict["vel"]

        return [batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel]

    def forward(self, x):
        head = self.parent.bbox_head
     
        pred_dict = self.head_forward(head, x, self.classes_eye)
        return self.get_bboxes(pred_dict, head.one_hot)


class CustomLayerNormImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps, x_shape):
        return F.layer_norm(input, normalized_shape, weight, bias, eps)

    @staticmethod
    def symbolic(g, input, normalized_shape, weight, bias, eps, x_shape):
        y = g.op("nv::CustomLayerNormalization", input, weight, bias, axis_i=-1, epsilon_f=eps)
        y.setType(input.type().with_sizes(x_shape))
        return y

class CustomLayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomLayerNormImpl.apply(
            input, self.normalized_shape, self.weight, self.bias, self.eps, input.size())

    @staticmethod
    def convert(old: nn.LayerNorm):
        Custom_layernorm = CustomLayerNorm(old.normalized_shape, old.eps, old.elementwise_affine)
        if Custom_layernorm.weight is not None:
            Custom_layernorm.weight.data = old.weight.data
            Custom_layernorm.bias.data   = old.bias.data
        return Custom_layernorm
    

class SubclassCameraModule(nn.Module):
    def __init__(self, model):
        super(SubclassCameraModule, self).__init__()
        self.model = model

    def forward(self, img, depth):
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        feat = self.model.img_backbone(img)
        feat = self.model.img_neck(feat)
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]

        BN, C, H, W = map(int, feat.size())
        feat = feat.view(B, int(BN / B), C, H, W)

        def get_cam_feats(self, x, d):
            B, N, C, fH, fW = map(int, x.shape)
            d = d.view(B * N, *d.shape[2:])
            x = x.view(B * N, C, fH, fW)

            d = self.dtransform(d)
            x = torch.cat([d, x], dim=1)
            x = self.depthnet(x)

            depth = x[:, : self.D].softmax(dim=1)
            feat  = x[:, self.D : (self.D + self.C)].permute(0, 2, 3, 1)
            return feat, depth

        return get_cam_feats(self.model.view_transform, feat, depth)

def replace_layernorm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent)
            setattr(parent, child, CustomLayerNorm.convert(module))


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
    # quantize.quantize_encoders_camera_branch(model)
    quantize.replace_to_quantization_module(model.fusion_layer)
    quantize.quantize_decoder(model.pts_backbone)
    quantize.quantize_decoder(model.pts_neck)
    model.pts_middle_encoder = funcs.layer_fusion_bn(model.pts_middle_encoder)
    return model




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

    return model, data_loader,


def qat(args):
    quantize.initialize()
    model, data_loader = init_model(args.config, args.checkpoint, device=args.device)
    model = quantize_net(model)
    model = fuse_conv_bn(model)
    print("🔥 start calibrate 🔥 ")
    quantize.set_quantizer_fast(model)
    quantize.calibrate_model(model, data_loader, 0, None, args.qat_batch_iter)

    quantize.disable_quantization(model.pts_middle_encoder.conv_input).apply()
    quantize.disable_quantization(model.pts_neck.deblocks[0][0]).apply()
    quantize.print_quantizer_status(model)
    print(f"Done due to ptq only! Save checkpoint to {args.qat_save_path} 🤗")
    model.pts_middle_encoder = funcs.fuse_relu_only(model.pts_middle_encoder)
    torch.save(model, args.qat_save_path)

def export_scn(args):
    inverse_indices = args.inverse
    if inverse_indices:
        scn_save = os.path.splitext(args.save)[0] + "/lidar.backbone.xyz.onnx"
    else:
        scn_save = os.path.splitext(args.save)[0] + "/lidar.backbone.xyz.onnx"
    
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model = torch.load(args.qat_save_path)
    model.eval().cuda().half()
    model = model.pts_middle_encoder

    quantize.disable_quantization(model).apply()

    # Set layer attributes
    for name, module in model.named_modules():
        module.precision = args.precision
        module.output_precision = args.precision
    
    model.conv_input.precision = "fp16"
    model.conv_out.output_precision = "fp16"

    voxels = torch.zeros(1, args.in_channel).cuda().half()
    coors  = torch.zeros(1, 4).int().cuda()
    batch_size = 1
    
    exptool.export_onnx(model, voxels, coors, batch_size, inverse_indices, scn_save)
    print(f"ONNX model has been saved to {scn_save}")

def export_neck_head(args):
    model = torch.load(args.qat_save_path)

    if args.precision == 'fp16':
        quantize.disable_quantization(model).apply()
    else:
        model.pts_neck.deblocks[1][0]._input_quantizer.disable()
        model.pts_neck.deblocks[1][0]._weight_quantizer.disable()

    fuser = SubclassFuser(model).cuda()
    head = SubclassHead(model).cuda()
    replace_layernorm(head)

    TensorQuantizer.use_fb_fake_quant = True
    with torch.no_grad():
        # camera_features = torch.randn(1, 80, 180, 180).cuda()
        # lidar_features  = torch.randn(1, 256, 180, 180).cuda()
        # lidar_features  = torch.randn(1, 256, 192, 120).cuda()
        bev_shape = [args.sparse_shape[0] // 8, args.sparse_shape[1] // 8]
        img_feature = torch.randn(1, args.img_inchannel, bev_shape[0], bev_shape[1]).cuda()
        lidar_features  = torch.randn(1, args.lidar_feature_channel, bev_shape[0], bev_shape[1]).cuda()

        fuser_onnx_path = f"{args.save}/fuser.onnx"
        torch.onnx.export(fuser, [img_feature, lidar_features], fuser_onnx_path, opset_version=13, 
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"🚀 The export is completed. ONNX save as {fuser_onnx_path} 🤗, Have a nice day~")


        boxhead_onnx_path = f"{args.save}/head.bbox.onnx"
        head_input = torch.randn(1, args.head_inchannel, bev_shape[0], bev_shape[1]).cuda()
        torch.onnx.export(head, head_input, f"{args.save}/head.bbox.onnx", opset_version=13, 
            input_names=["middle"],
            output_names=["score", "rot", "dim", "reg", "height", "vel"],
        )
        print(f"🚀 The export is completed. ONNX save as {boxhead_onnx_path} 🤗, Have a nice day~")

    print("done")

def export_camera(args):
    model = torch.load(args.qat_save_path)
    quantize.disable_quantization(model).apply()
    camera_model = SubclassCameraModule(model)
    camera_model.cuda().eval()
    downsample_model = model.view_transform.downsample
    downsample_model.cuda().eval()

    # points = [torch.load('projects/BEVFusion/qat/input_data/points.pt').cuda()]
    # imgs = torch.load('projects/BEVFusion/qat/input_data/imgs.pt').cuda()

    points = [torch.load('projects/BEVFusion/qat/ns_input_data/points.pt').cuda()]
    imgs = torch.load('projects/BEVFusion/qat/ns_input_data/imgs.pt').cuda()



    depth = torch.zeros(len(points), imgs.shape[1], 1, imgs.shape[-2], imgs.shape[-1]).cuda()
    downsample_model = model.view_transform.downsample
    downsample_model.cuda().eval()
    bev_shape = [args.sparse_shape[0] // 8, args.sparse_shape[1] // 8]
    downsample_in = torch.zeros(1, 80, bev_shape[0] * 2, bev_shape[1] * 2).cuda()

    save_root = os.path.splitext(args.save)[0]
    os.makedirs(save_root, exist_ok=True)
    with torch.no_grad():
        camera_backbone_onnx = f"{save_root}/camera.backbone.onnx"
        camera_vtransform_onnx = f"{save_root}/camera.vtransform.onnx"
        TensorQuantizer.use_fb_fake_quant = True
        torch.onnx.export(
            camera_model,
            (imgs, depth),
            camera_backbone_onnx,
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )

        onnx_orig = onnx.load(camera_backbone_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, camera_backbone_onnx)
        print(f"🚀 The export is completed. ONNX save as {camera_backbone_onnx} 🤗, Have a nice day~")

        torch.onnx.export(
            downsample_model,
            downsample_in,
            camera_vtransform_onnx,
            input_names=["feat_in"],
            output_names=["feat_out"],
            opset_version=13,
            do_constant_folding=True,
        )
        print(f"🚀 The export is completed. ONNX save as {camera_vtransform_onnx} 🤗, Have a nice day~")



def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Quantization Aware Training")
    parser.add_argument('--precision', type=str, default='fp16', help='precision control')
    parser.add_argument('--config', type=str, default='projects/BEVFusion/configs/bevfusion_lidar_camera_yx_kl.py', help='Config file')
    parser.add_argument('--checkpoint', type=str, default='work_dirs/bevfusion_lidar_camera_yx_kl/epoch_6.pth', help='Checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference')
    parser.add_argument('--qat_save_path', type=str, default='projects/BEVFusion/qat/ckpt/ptq_bevfusion.pth', help='Path to save the quantized model')
    parser.add_argument('--qat_batch_iter', type=int, default=500, help='iter to calibrate the quantized model')
    parser.add_argument('--save', type=str, default='projects/BEVFusion/qat/onnx', help='Path to save the quantized model')
    parser.add_argument('--inverse', type=bool, default=False, help='voxel xyz or zxy')
    return EasyDict(vars(parser.parse_args()))


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)
    args.in_channel = config.model.pts_middle_encoder.in_channels
    args.sparse_shape = config.model.pts_middle_encoder.sparse_shape
    args.lidar_feature_channel = config.model.pts_neck.out_channels[1]

    args.head_inchannel = config.model.bbox_head.in_channels
    args.img_inchannel = config.model.fusion_layer.in_channels[0]
    qat(args)
    
    export_camera(args)
    export_scn(args)
    export_neck_head(args)


# python projects/BEVFusion/qat/qat_export_model.py --precision int8 --config work_dirs/bevfusion_yx_kl/bevfusion_yx_kl.py --checkpoint work_dirs/bevfusion_yx_kl/epoch_16.pth
