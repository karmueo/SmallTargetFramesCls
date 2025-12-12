from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-18 特征提取器，移除分类头并支持灰度/彩色输入。

    Args:
        pretrained: 是否加载 ImageNet 预训练权重。
        in_channels: 输入通道数，支持 1 或 3。
        global_pool: 使用的全局池化方式，"avg" 或 "max"。
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        global_pool: Literal["avg", "max"] = "avg",
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if global_pool not in {"avg", "max"}:
            raise ValueError(f"Unsupported global_pool: {global_pool}")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.conv1 = self._build_input_conv(
            original_conv=backbone.conv1,
            in_channels=in_channels,
            has_pretrained=weights is not None,
        )
        self._set_layer_stride_to_one(backbone.layer3)
        self._set_layer_stride_to_one(backbone.layer4)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if global_pool == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self.out_channels = backbone.fc.in_features

    @staticmethod
    def _build_input_conv(
        original_conv: nn.Conv2d, in_channels: int, has_pretrained: bool
    ) -> nn.Conv2d:
        """构造与目标通道数匹配的输入卷积，必要时复用预训练权重。"""
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=original_conv.bias is not None,
        )
        if original_conv.bias is not None:
            new_conv.bias.data.copy_(original_conv.bias.data)

        if has_pretrained:
            with torch.no_grad():
                weight = original_conv.weight
                if weight.shape[2:] != new_conv.weight.shape[2:]:
                    weight = F.interpolate(
                        weight, size=new_conv.weight.shape[2:], mode="bilinear", align_corners=False
                    )
                if in_channels == original_conv.in_channels:
                    new_conv.weight.copy_(weight)
                elif in_channels == 1:
                    new_conv.weight.copy_(weight.mean(dim=1, keepdim=True))
                else:
                    expanded = weight.mean(dim=1, keepdim=True)
                    expanded = expanded.repeat(1, in_channels, 1, 1)
                    new_conv.weight.copy_(expanded)
        return new_conv

    @staticmethod
    def _set_layer_stride_to_one(layer: nn.Sequential) -> None:
        """将层的首个块下采样去除，保持空间分辨率。"""
        block = layer[0]
        if hasattr(block, "stride"):
            block.stride = 1
        if hasattr(block, "conv1") and isinstance(block.conv1, nn.Conv2d):
            block.conv1.stride = (1, 1)
        if hasattr(block, "downsample") and block.downsample is not None:
            for mod in block.downsample:
                if isinstance(mod, nn.Conv2d):
                    mod.stride = (1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (B, C, H, W)。

        Returns:
            特征向量，形状为 (B, out_channels)。
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def build_feature_extractor(
    name: str = "resnet18",
    pretrained: bool = False,
    in_channels: int = 3,
    global_pool: Literal["avg", "max"] = "avg",
) -> nn.Module:
    """
    根据名称构建特征提取器。

    Args:
        name: 模型名称，支持 "resnet18"、"mobilenet_v2"。
        pretrained: 是否加载预训练权重。
        in_channels: 输入通道数。
        global_pool: 全局池化方式，"avg" 或 "max"。
    """
    name_lower = name.lower()
    if name_lower in {"resnet18", "resnet-18"}:
        return ResNetFeatureExtractor(
            pretrained=pretrained, in_channels=in_channels, global_pool=global_pool
        )
    if name_lower in {"mobilenet_v2", "mobilenetv2"}:
        return MobileNetV2FeatureExtractor(
            pretrained=pretrained, in_channels=in_channels, global_pool=global_pool
        )
    raise ValueError(f"Unsupported feature extractor: {name}")


class MobileNetV2FeatureExtractor(nn.Module):
    """
    MobileNetV2 特征提取器，调整 stem stride 以保留分辨率，并支持灰度/彩色输入。

    Args:
        pretrained: 是否加载 ImageNet 预训练权重。
        in_channels: 输入通道数。
        global_pool: 使用的全局池化方式。
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        global_pool: Literal["avg", "max"] = "avg",
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if global_pool not in {"avg", "max"}:
            raise ValueError(f"Unsupported global_pool: {global_pool}")

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        # stem: ConvBNReLU(3,32,kernel=3,stride=2) -> stride 改为 1，适配通道
        conv: nn.Conv2d = backbone.features[0][0]
        backbone.features[0][0] = self._build_input_conv(
            original_conv=conv,
            in_channels=in_channels,
            has_pretrained=weights is not None,
        )

        self.features = backbone.features
        self.pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if global_pool == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self.out_channels = backbone.classifier[1].in_features

    @staticmethod
    def _build_input_conv(
        original_conv: nn.Conv2d, in_channels: int, has_pretrained: bool
    ) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=original_conv.bias is not None,
        )
        if original_conv.bias is not None:
            new_conv.bias.data.copy_(original_conv.bias.data)

        if has_pretrained:
            with torch.no_grad():
                weight = original_conv.weight
                if weight.shape[2:] != new_conv.weight.shape[2:]:
                    weight = F.interpolate(
                        weight,
                        size=new_conv.weight.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if in_channels == original_conv.in_channels:
                    new_conv.weight.copy_(weight)
                elif in_channels == 1:
                    new_conv.weight.copy_(weight.mean(dim=1, keepdim=True))
                else:
                    expanded = weight.mean(dim=1, keepdim=True).repeat(
                        1, in_channels, 1, 1
                    )
                    new_conv.weight.copy_(expanded)
        return new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (B, C, H, W)。

        Returns:
            特征向量，形状为 (B, out_channels)。
        """
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
