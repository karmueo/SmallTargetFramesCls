from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch
from torch import nn

from src.model_cnn_tcn import TemporalConvNet
from .feature_extractor import build_feature_extractor
from .position_encoder import PositionEncoder, PositionEncoderConfig


@dataclass(frozen=True)
class ClassifierConfig:
    """分类器高层配置。"""

    feature_name: str = "resnet18"
    pretrained: bool = False
    in_channels: int = 3
    global_pool: Literal["avg", "max"] = "avg"
    tcn_channels: Sequence[int] = (128, 128)
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1
    temporal_pool: Literal["last", "mean", "max"] = "last"
    num_classes: int = 4
    head_dropout: float = 0.0
    use_position: bool = False
    position_embed_dim: int = 64
    position_hidden_dim: int | None = 64
    position_dropout: float = 0.0
    position_layernorm: bool = True


class TemporalClassifier(nn.Module):
    """
    融合 CNN 特征与 TCN 的时序分类器。

    输入形状为 (B, T, C, H, W)，先逐帧提取特征，再经 TCN 建模时序，最终池化到序列级别做分类。
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        num_classes: int,
        tcn_channels: Sequence[int] | Iterable[int],
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        temporal_pool: Literal["last", "mean", "max"] = "last",
        head_dropout: float = 0.0,
        position_encoder: PositionEncoder | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if tcn_kernel_size < 1 or tcn_kernel_size % 2 == 0:
            raise ValueError("tcn_kernel_size must be odd and >=1")
        if temporal_pool not in {"last", "mean", "max"}:
            raise ValueError(f"Unsupported temporal_pool: {temporal_pool}")

        self.feature_extractor = feature_extractor
        feature_dim = self._infer_feature_dim(feature_extractor)
        tcn_channels = list(tcn_channels)
        if not tcn_channels:
            raise ValueError("tcn_channels must be non-empty")

        self.position_encoder = position_encoder
        if self.position_encoder is not None:
            feature_dim += self.position_encoder.config.embed_dim

        self.tcn = TemporalConvNet(
            num_inputs=feature_dim,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )
        self.temporal_pool = temporal_pool
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(tcn_channels[-1], num_classes)

    @staticmethod
    def _infer_feature_dim(feature_extractor: nn.Module) -> int:
        """优先从 out_channels 推断特征维度，否则尝试 fc.in_features。"""
        if hasattr(feature_extractor, "out_channels"):
            dim = getattr(feature_extractor, "out_channels")
            if isinstance(dim, int) and dim > 0:
                return dim
        if hasattr(feature_extractor, "fc") and hasattr(feature_extractor.fc, "in_features"):
            dim = int(feature_extractor.fc.in_features)  # type: ignore[attr-defined]
            if dim > 0:
                return dim
        raise ValueError("Cannot infer feature dimension from feature_extractor")

    def _pool_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """根据策略对时序维度做池化。输入形状 (B, C, T)。"""
        if self.temporal_pool == "last":
            return x[:, :, -1]
        if self.temporal_pool == "mean":
            return x.mean(dim=2)
        return x.max(dim=2).values

    def forward(self, images: torch.Tensor, boxes: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            images: 输入张量，形状 (B, T, C, H, W)。
            boxes: 可选，形状 (B, T, 4) 的归一化 bbox (cx, cy, w, h)。

        Returns:
            包含 `logits` (B, num_classes) 以及 `features` (B, C_tcn) 的字典。
        """
        if images.dim() != 5:
            raise ValueError(f"Expected images with 5 dims (B,T,C,H,W), got {images.shape}")
        b, t, c, h, w = images.shape
        x = images.view(b * t, c, h, w)
        frame_feats = self.feature_extractor(x)
        frame_feats = frame_feats.view(b, t, -1)

        if self.position_encoder is not None:
            if boxes is None:
                raise ValueError("boxes must be provided when position_encoder is enabled")
            pos_emb = self.position_encoder(boxes)
            frame_feats = torch.cat([frame_feats, pos_emb], dim=-1)

        frame_feats = frame_feats.transpose(1, 2)  # (B, C_feat(+pos), T)

        seq_feats = self.tcn(frame_feats)  # (B, C_tcn, T)
        pooled = self._pool_temporal(seq_feats)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits, "features": pooled}


def build_classifier(config: ClassifierConfig) -> TemporalClassifier:
    """
    根据配置构建完整分类器。

    Args:
        config: 分类器配置。
    """
    feature_extractor = build_feature_extractor(
        name=config.feature_name,
        pretrained=config.pretrained,
        in_channels=config.in_channels,
        global_pool=config.global_pool,
    )
    position_encoder = None
    if config.use_position:
        position_encoder = PositionEncoder(
            PositionEncoderConfig(
                embed_dim=config.position_embed_dim,
                hidden_dim=config.position_hidden_dim,
                dropout=config.position_dropout,
                use_layernorm=config.position_layernorm,
            )
        )
    return TemporalClassifier(
        feature_extractor=feature_extractor,
        num_classes=config.num_classes,
        tcn_channels=config.tcn_channels,
        tcn_kernel_size=config.tcn_kernel_size,
        tcn_dropout=config.tcn_dropout,
        temporal_pool=config.temporal_pool,
        head_dropout=config.head_dropout,
        position_encoder=position_encoder,
    )
