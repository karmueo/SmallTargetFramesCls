from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts import train as train_module
from src.data import FrameSamplingConfig, SequenceDataset


def build_test_loader(cfg: Dict, root_override: Path | None = None, test_csv_override: Path | None = None) -> Tuple[DataLoader, Dict[str, int]]:
    data_cfg = cfg["data"].copy()
    if root_override is not None:
        data_cfg["root"] = str(root_override)
    if test_csv_override is not None:
        data_cfg["test_csv"] = str(test_csv_override)
    if "test_csv" not in data_cfg:
        raise ValueError("test_csv must be provided in config or override")

    sampling_cfg = FrameSamplingConfig(**data_cfg["frame_sampling"])
    class_to_idx = train_module.load_class_map(Path(data_cfg["class_map"]))

    image_transform = train_module.build_transforms(
        image_size=int(data_cfg["image_size"]),
        mean=data_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=data_cfg.get("std", [0.229, 0.224, 0.225]),
    )

    test_ds = SequenceDataset(
        csv_path=Path(data_cfg["test_csv"]),
        root_dir=Path(data_cfg["root"]),
        sampling=sampling_cfg,
        image_transform=image_transform,
        return_tensor=True,
        skip_broken=False,
        class_to_idx=class_to_idx,
    )

    loader = DataLoader(
        test_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=train_module.collate_batch,
    )
    return loader, class_to_idx


def load_model(cfg: Dict, class_to_idx: Dict[str, int], checkpoint: Path, device: torch.device) -> nn.Module:
    model = train_module.build_model(cfg, num_classes=len(class_to_idx), device=device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model_state", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        boxes = batch.get("boxes")
        if boxes is not None:
            boxes = boxes.to(device, non_blocking=True)
        outputs = model(images, boxes=boxes)
        logits = outputs["logits"]
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "samples": float(total),
    }


def run_eval(cfg: Dict, checkpoint: Path, root: Path | None = None, test_csv: Path | None = None, device_str: str | None = None) -> Dict[str, float]:
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, class_to_idx = build_test_loader(cfg, root_override=root, test_csv_override=test_csv)
    model = load_model(cfg, class_to_idx=class_to_idx, checkpoint=checkpoint, device=device)
    metrics = evaluate(model, loader, device)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classifier on test split")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to YAML config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file (e.g., checkpoints/best.pt)")
    parser.add_argument("--root", type=Path, default=None, help="Override dataset root")
    parser.add_argument("--test-csv", type=Path, default=None, help="Override test CSV path")
    parser.add_argument("--device", type=str, default=None, help='Device string, e.g., "cpu" or "cuda:0"')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = train_module.load_config(args.config)
    metrics = run_eval(cfg, checkpoint=args.checkpoint, root=args.root, test_csv=args.test_csv, device_str=args.device)
    print(f"Test accuracy: {metrics['accuracy']:.4f}, loss: {metrics['loss']:.4f}, samples: {int(metrics['samples'])}")


if __name__ == "__main__":
    main()
