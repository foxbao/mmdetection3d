# modify from https://github.com/mit-han-lab/bevfusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class GeneralizedLSSFPN(BaseModule):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            no_norm_on_lateral=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(mode='bilinear', align_corners=True),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i] +
                (in_channels[i + 1] if i == self.backbone_end_level -
                 1 else out_channels),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)


class BasicBlock(nn.Module):
    """Basic residual block for GeneralizedResNet."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


@MODELS.register_module()
class GeneralizedResNet(BaseModule):
    """Generalized ResNet backbone for BEV features.

    Ref: https://github.com/mit-han-lab/bevfusion

    Args:
        in_channels (int): Input channels (e.g., 80 for camera BEV).
        blocks (list): List of [num_blocks, out_channels, stride] per stage.
    """

    def __init__(self, in_channels, blocks):
        super().__init__()
        self.blocks = blocks
        self.stages = nn.ModuleList()
        current_channels = in_channels
        for num_blocks, out_channels, stride in blocks:
            layers = []
            layers.append(BasicBlock(current_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(BasicBlock(out_channels, out_channels, 1))
            self.stages.append(nn.Sequential(*layers))
            current_channels = out_channels

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


@MODELS.register_module()
class LSSFPN(BaseModule):
    """LSS-style FPN for BEV features.

    Ref: https://github.com/mit-han-lab/bevfusion

    Args:
        in_channels (list[int]): Input channels from selected stages.
        in_indices (list[int]): Indices of stages to use.
        out_channels (int): Output channels.
        scale_factor (int): Upsample scale factor.
    """

    def __init__(self, in_channels, in_indices, out_channels, scale_factor=2):
        super().__init__()
        self.in_indices = in_indices
        self.fuse = nn.Sequential(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scale_factor = scale_factor

    def forward(self, inputs):
        # inputs is a list of feature maps from GeneralizedResNet
        target = inputs[self.in_indices[0]]
        feats = []
        for idx in self.in_indices:
            feat = inputs[idx]
            if feat.shape[2:] != target.shape[2:]:
                feat = F.interpolate(
                    feat, size=target.shape[2:],
                    mode='bilinear', align_corners=True)
            feats.append(feat)
        x = self.fuse(torch.cat(feats, dim=1))
        if self.scale_factor != 1:
            x = F.interpolate(
                x, scale_factor=self.scale_factor,
                mode='bilinear', align_corners=True)
        return x
