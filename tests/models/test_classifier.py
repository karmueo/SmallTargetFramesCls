import pytest

torch = pytest.importorskip("torch")

from src.models import ClassifierConfig, ResNetFeatureExtractor, TemporalClassifier, build_classifier


def test_temporal_classifier_forward_last_pool():
    feature_extractor = ResNetFeatureExtractor(pretrained=False, in_channels=3)
    model = TemporalClassifier(
        feature_extractor=feature_extractor,
        num_classes=5,
        tcn_channels=(64, 64),
        tcn_kernel_size=3,
        temporal_pool="last",
        head_dropout=0.0,
    )
    x = torch.randn(2, 4, 3, 64, 64)

    outputs = model(x)

    assert set(outputs.keys()) == {"logits", "features"}
    assert outputs["logits"].shape == (2, 5)
    assert outputs["features"].shape == (2, 64)


def test_build_classifier_grayscale_mean_pool():
    config = ClassifierConfig(
        in_channels=1,
        num_classes=3,
        temporal_pool="mean",
        tcn_channels=(32,),
        pretrained=False,
    )
    model = build_classifier(config)
    x = torch.randn(1, 5, 1, 64, 64)

    outputs = model(x)

    assert outputs["logits"].shape == (1, 3)
    assert outputs["features"].shape[1] == config.tcn_channels[-1]
    assert torch.isfinite(outputs["logits"]).all()


def test_classifier_with_position_encoder():
    config = ClassifierConfig(
        use_position=True,
        position_embed_dim=8,
        position_hidden_dim=16,
        num_classes=2,
        tcn_channels=(32,),
        in_channels=3,
        pretrained=False,
    )
    model = build_classifier(config)
    images = torch.randn(1, 3, 3, 64, 64)
    boxes = torch.rand(1, 3, 4)

    outputs = model(images, boxes=boxes)
    assert outputs["logits"].shape == (1, 2)
    assert outputs["features"].shape == (1, config.tcn_channels[-1])
