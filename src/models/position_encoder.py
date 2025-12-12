from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def _build_geom_and_motion_features(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    将归一化 bbox (cx, cy, w, h) 转换为几何 + 运动特征。

    Args:
        boxes: 形状 (B, T, 4) 或 (T, 4)，值应在 [0,1]。
        eps: 数值稳定项。
    Returns:
        特征张量形状 (B, T, D) 或 (T, D)，D=14。
    """
    if boxes.dim() not in (2, 3) or boxes.size(-1) != 4:
        raise ValueError(f"boxes must have shape (..., T, 4), got {boxes.shape}")

    # 展开批次维度以统一处理
    orig_shape = boxes.shape
    if boxes.dim() == 2:
        boxes = boxes.unsqueeze(0)
    cx, cy, w, h = boxes.unbind(dim=-1)  # (B, T)

    ar = torch.log((w + eps) / (h + eps))
    area = w * h

    # 速度与尺度变化（第一帧补零）
    zeros = torch.zeros_like(cx[:, :1])
    vx = torch.cat([zeros, cx[:, 1:] - cx[:, :-1]], dim=1)
    vy = torch.cat([zeros, cy[:, 1:] - cy[:, :-1]], dim=1)
    vs = torch.cat([zeros, w[:, 1:] - w[:, :-1]], dim=1)
    vh = torch.cat([zeros, h[:, 1:] - h[:, :-1]], dim=1)

    # 加速度（首两帧补零）
    zeros2 = torch.zeros_like(cx[:, :1])
    ax = torch.cat([zeros2, vx[:, 1:] - vx[:, :-1]], dim=1)
    ay = torch.cat([zeros2, vy[:, 1:] - vy[:, :-1]], dim=1)

    speed = torch.sqrt(vx**2 + vy**2 + eps)
    scale_rate = torch.cat([zeros, (w[:, 1:] - w[:, :-1]) / (w[:, :-1] + eps)], dim=1)

    feats = torch.stack(
        [
            cx,
            cy,
            w,
            h,
            ar,
            area,
            vx,
            vy,
            vs,
            vh,
            ax,
            ay,
            speed,
            scale_rate,
        ],
        dim=-1,
    )  # (B, T, 14)
    if orig_shape == boxes.shape:
        return feats
    return feats.squeeze(0)


@dataclass(frozen=True)
class PositionEncoderConfig:
    """位置/运动特征编码配置。"""

    input_dim: int = 14
    embed_dim: int = 64
    hidden_dim: Optional[int] = 64
    dropout: float = 0.0
    use_layernorm: bool = True


class PositionEncoder(nn.Module):
    """
    将归一化 bbox 序列编码为嵌入向量，包含几何 + 运动特征。
    输入 boxes 形状 (B, T, 4)，输出 (B, T, embed_dim)。
    """

    def __init__(self, config: PositionEncoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.input_dim
        if config.hidden_dim is not None and config.hidden_dim > 0:
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            if config.use_layernorm:
                layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, config.embed_dim))
        if config.use_layernorm:
            layers.append(nn.LayerNorm(config.embed_dim))
        layers.append(nn.Dropout(config.dropout))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.config = config

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: 形状 (B, T, 4)，归一化 bbox (cx, cy, w, h)。

        Returns:
            嵌入张量 (B, T, embed_dim)。
        """
        feats = _build_geom_and_motion_features(boxes)  # (B, T, 14)
        return self.mlp(feats)
