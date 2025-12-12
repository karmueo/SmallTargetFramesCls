import sys
import json
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

# 确保仓库根路径可导入 scripts 与 src
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval as eval_module  # noqa: E402
from scripts import train as train_module  # noqa: E402


def _make_dummy_sequence(root: Path, cls: str, seq: str, num_frames: int = 2, size: int = 32) -> None:
    seq_dir = root / cls / seq
    seq_dir.mkdir(parents=True, exist_ok=True)
    tracks = []
    for i in range(num_frames):
        img_path = seq_dir / f"{i:08d}.jpg"
        Image.new("RGB", (size, size), color=(i * 10 % 255, 0, 0)).save(img_path)
        tracks.append(
            {
                "image": img_path.name,
                "bbox": {"x": 1, "y": 1, "w": size // 2, "h": size // 2},
            }
        )
    meta = {"resolution": {"width": size, "height": size}, "tracks": tracks}
    meta_path = seq_dir.parent / f"{seq_dir.name}.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _build_eval_config(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    _make_dummy_sequence(data_root, "bird", "seq_0", num_frames=2, size=32)

    test_csv = tmp_path / "test.csv"
    test_csv.write_text("bird/seq_0,bird\n", encoding="utf-8")

    class_map = tmp_path / "class_map.txt"
    class_map.write_text("0 bird\n", encoding="utf-8")

    return {
        "seed": 0,
        "data": {
            "root": str(data_root),
            "test_csv": str(test_csv),
            "class_map": str(class_map),
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "frame_sampling": {
                "target_length": 2,
                "strategy": "uniform",
                "sampling_rate": 1,
                "pad_mode": "repeat",
                "jitter": False,
                "seed": 0,
            },
            "image_size": 32,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "model": {
            "feature_name": "resnet18",
            "pretrained": False,
            "in_channels": 3,
            "global_pool": "avg",
            "tcn_channels": [16],
            "tcn_kernel_size": 3,
            "tcn_dropout": 0.0,
            "temporal_pool": "last",
            "head_dropout": 0.0,
            "num_classes": 1,
        },
        "optimizer": {"name": "adam", "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "scheduler": {"type": "none", "warmup_steps": 0},
        "training": {
            "epochs": 1,
            "log_interval": 1,
            "val_interval": 1,
            "grad_clip_norm": 0.0,
            "mixed_precision": False,
            "ckpt_dir": str(tmp_path / "ckpts"),
            "save_best": True,
            "resume": None,
        },
    }


def test_run_eval_with_checkpoint(tmp_path):
    cfg = _build_eval_config(tmp_path)
    device = torch.device("cpu")

    # 构造恒为类别0的模型 checkpoint
    model = train_module.build_model(cfg, num_classes=1, device=device)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    metrics = eval_module.run_eval(cfg, checkpoint=ckpt_path, device_str="cpu")
    assert metrics["accuracy"] == 1.0
    assert metrics["samples"] == 1.0
