import torch

from src.models.position_encoder import PositionEncoder, PositionEncoderConfig, _build_geom_and_motion_features


def test_geom_and_motion_feature_shapes_and_zeros():
    boxes = torch.tensor(
        [
            [
                [0.5, 0.5, 0.2, 0.1],  # t0
                [0.6, 0.55, 0.25, 0.12],  # t1
            ]
        ]
    )
    feats = _build_geom_and_motion_features(boxes)
    assert feats.shape == (1, 2, 14)
    # 第一帧速度/加速度应为 0
    assert torch.allclose(feats[:, 0, 6:12], torch.zeros_like(feats[:, 0, 6:12]))


def test_position_encoder_forward():
    config = PositionEncoderConfig(embed_dim=16, hidden_dim=32, dropout=0.0, use_layernorm=True)
    encoder = PositionEncoder(config)
    boxes = torch.rand(2, 4, 4)

    out = encoder(boxes)
    assert out.shape == (2, 4, config.embed_dim)
