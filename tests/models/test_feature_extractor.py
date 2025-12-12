import pytest

torch = pytest.importorskip("torch")

from src.models import ResNetFeatureExtractor


def test_resnet_feature_extractor_rgb_forward_shape():
    model = ResNetFeatureExtractor(pretrained=False, in_channels=3)
    x = torch.randn(2, 3, 64, 64)

    out = model(x)

    assert out.shape == (2, model.out_channels)


def test_resnet_feature_extractor_grayscale_support():
    model = ResNetFeatureExtractor(pretrained=False, in_channels=1)
    x = torch.randn(1, 1, 64, 64)

    out = model(x)

    # 确认输入卷积改为单通道且输出维度正确
    conv1 = model.stem[0]
    assert conv1.weight.shape[1] == 1
    assert out.shape == (1, model.out_channels)
