import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

# 确保仓库根路径可导入 scripts 与 src
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import export_onnx as export_module  # noqa: E402


class _DummyModel(torch.nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, images: torch.Tensor, boxes: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # images: (B, T, C, H, W)
        x = images.mean(dim=1)  # (B, C, H, W)
        x = x.flatten(1)
        x = self.dropout(x)
        logits = self.linear(x)
        return {"logits": logits}


def test_export_to_onnx_restores_eval_mode(tmp_path: Path) -> None:
    model = _DummyModel().eval()
    images = torch.randn(2, 3, 3, 8, 8)

    onnx_path = tmp_path / "dummy.onnx"
    export_module.export_to_onnx(model, images, boxes=None, output_path=onnx_path, opset=17)

    assert model.training is False
    assert all(m.training is False for m in model.modules() if isinstance(m, torch.nn.Dropout))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pt_logits = model(images)["logits"].detach()
    ort_logits = torch.from_numpy(session.run(None, {"images": images.numpy()})[0])
    diff = (pt_logits - ort_logits).abs()
    assert diff.max().item() < 1e-3

