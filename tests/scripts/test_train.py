import sys
from pathlib import Path

import json
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

# 确保仓库根路径可导入 scripts 与 src
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _build_minimal_config(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    _make_dummy_sequence(data_root, "bird", "seq_0", num_frames=2, size=32)
    _make_dummy_sequence(data_root, "uav", "seq_1", num_frames=2, size=32)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_csv.write_text("bird/seq_0,bird\nuav/seq_1,uav\n", encoding="utf-8")
    val_csv.write_text("bird/seq_0,bird\n", encoding="utf-8")

    class_map = tmp_path / "class_map.txt"
    class_map.write_text("0 bird\n1 uav\n", encoding="utf-8")

    return {
        "seed": 0,
        "data": {
            "root": str(data_root),
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "class_map": str(class_map),
            "batch_size": 2,
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
            "tcn_channels": [32],
            "tcn_kernel_size": 3,
            "tcn_dropout": 0.0,
            "temporal_pool": "last",
            "head_dropout": 0.0,
            "num_classes": 2,
        },
        "optimizer": {
            "name": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
        },
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 2,
            "step_size_epochs": 1,
            "gamma": 0.5,
        },
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


def test_load_config_and_class_map(tmp_path):
    cfg_dict = {"foo": "bar"}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(train_module.yaml.safe_dump(cfg_dict), encoding="utf-8")
    loaded = train_module.load_config(cfg_path)
    assert loaded == cfg_dict

    class_map_path = tmp_path / "class_map.txt"
    class_map_path.write_text("0 a\n1 b\n", encoding="utf-8")
    mapping = train_module.load_class_map(class_map_path)
    assert mapping == {"a": 0, "b": 1}


def test_build_dataloaders_and_scheduler(tmp_path):
    cfg = _build_minimal_config(tmp_path)
    device = torch.device("cpu")
    train_loader, val_loader, class_to_idx = train_module.build_dataloaders(cfg, device)

    batch = next(iter(train_loader))
    assert batch["images"].shape == (2, 2, 3, 32, 32)
    assert batch["labels"].shape == (2,)
    assert class_to_idx == {"bird": 0, "uav": 1}

    optimizer = train_module.build_optimizer(
        torch.nn.Linear(4, 2), cfg["optimizer"]
    )
    scheduler = train_module.build_scheduler(
        optimizer, cfg["scheduler"], total_steps=10, steps_per_epoch=len(train_loader)
    )
    assert scheduler is not None
    lrs = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    warmup = cfg["scheduler"]["warmup_steps"]
    decay_part = lrs[warmup:] if warmup < len(lrs) else lrs
    assert decay_part[-1] <= decay_part[0]  # warmup 后应衰减或持平


def test_train_loop_runs_one_epoch(tmp_path):
    cfg = _build_minimal_config(tmp_path)
    cfg["training"]["epochs"] = 1
    cfg["training"]["mixed_precision"] = False
    cfg["training"]["ckpt_dir"] = str(tmp_path / "ckpts")
    train_module.train_loop(cfg)

    ckpt_dir = Path(cfg["training"]["ckpt_dir"])
    assert (ckpt_dir / "epoch_1.pt").is_file()
    assert (ckpt_dir / "best.pt").is_file()
