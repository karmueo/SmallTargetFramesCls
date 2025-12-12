from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import FrameSamplingConfig, SequenceDataset
from src.models import ClassifierConfig, build_classifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_class_map(path: Path) -> Dict[str, int]:
    """读取 class_map.txt，格式：index label"""
    if not path.is_file():
        raise FileNotFoundError(f"class_map file not found: {path}")
    mapping: Dict[str, int] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            label = parts[1]
            mapping[label] = idx
    if not mapping:
        raise ValueError(f"No classes parsed from {path}")
    return mapping


def build_transforms(image_size: int, mean: List[float], std: List[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    images = torch.stack([sample["images"] for sample in batch], dim=0)  # (B, T, C, H, W)
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    boxes = torch.stack([sample["boxes"] for sample in batch], dim=0)  # (B, T, 4)
    return {"images": images, "labels": labels, "boxes": boxes}


def build_dataloaders(cfg: Dict, device: torch.device) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    data_cfg = cfg["data"]
    sampling_cfg = FrameSamplingConfig(**data_cfg["frame_sampling"])

    class_map_path = Path(data_cfg.get("class_map", ""))
    class_to_idx = load_class_map(class_map_path) if class_map_path else None

    image_transform = build_transforms(
        image_size=int(data_cfg["image_size"]),
        mean=data_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=data_cfg.get("std", [0.229, 0.224, 0.225]),
    )

    train_ds = SequenceDataset(
        csv_path=Path(data_cfg["train_csv"]),
        root_dir=Path(data_cfg["root"]),
        sampling=sampling_cfg,
        image_transform=image_transform,
        return_tensor=True,
        skip_broken=False,
        class_to_idx=class_to_idx,
    )
    val_ds = SequenceDataset(
        csv_path=Path(data_cfg["val_csv"]),
        root_dir=Path(data_cfg["root"]),
        sampling=sampling_cfg,
        image_transform=image_transform,
        return_tensor=True,
        skip_broken=False,
        class_to_idx=train_ds.class_to_idx,
    )

    common_kwargs = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "collate_fn": collate_batch,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader, train_ds.class_to_idx


def build_model(cfg: Dict, num_classes: int, device: torch.device) -> nn.Module:
    model_cfg = cfg["model"].copy()
    model_cfg["num_classes"] = num_classes
    classifier_config = ClassifierConfig(**model_cfg)
    model = build_classifier(classifier_config)
    return model.to(device)


def build_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    total_steps: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_type = cfg.get("type", "none").lower()
    warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
    if sched_type == "none" and warmup_steps <= 0:
        return None

    step_size_epochs = max(int(cfg.get("step_size_epochs", 1)), 1)
    step_size_steps = step_size_epochs * max(steps_per_epoch, 1)
    gamma = float(cfg.get("gamma", 0.1))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        adjusted_step = current_step - warmup_steps
        if adjusted_step < 0:
            adjusted_step = 0
        if sched_type == "cosine":
            if total_steps <= warmup_steps:
                return 1.0
            progress = adjusted_step / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if sched_type == "step":
            if step_size_steps <= 0:
                return 1.0
            return gamma ** (adjusted_step // step_size_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
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
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    best_acc: float,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_acc": best_acc,
            "config": config,
        },
        path,
    )


def train_loop(cfg: Dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    train_loader, val_loader, class_to_idx = build_dataloaders(cfg, device)
    num_classes = len(class_to_idx)
    model = build_model(cfg, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg["optimizer"])

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * int(cfg["training"]["epochs"])
    scheduler = build_scheduler(
        optimizer,
        cfg["scheduler"],
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
    )
    scaler = GradScaler(enabled=bool(cfg["training"].get("mixed_precision", True)))

    start_epoch = 0
    best_acc = 0.0
    resume_path = cfg["training"].get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", 0.0))
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    log_interval = int(cfg["training"].get("log_interval", 10))
    val_interval = int(cfg["training"].get("val_interval", 1))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 0.0) or 0.0)
    ckpt_dir = Path(cfg["training"].get("ckpt_dir", "checkpoints"))
    save_best = bool(cfg["training"].get("save_best", True))

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            boxes = batch.get("boxes")
            if boxes is not None:
                boxes = boxes.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                outputs = model(images, boxes=boxes)
                logits = outputs["logits"]
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            global_step += 1

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / max(running_total, 1)
                acc = running_correct / max(running_total, 1)
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch+1}/{cfg['training']['epochs']}], "
                    f"Step [{batch_idx+1}/{steps_per_epoch}], "
                    f"LR {current_lr:.3e}, Loss {avg_loss:.4f}, Acc {acc:.4f}"
                )
                running_loss = 0.0
                running_correct = 0
                running_total = 0

        if (epoch + 1) % val_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            print(f"Validation - Epoch {epoch+1}: loss {val_loss:.4f}, acc {val_acc:.4f}")
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, best_acc, cfg)
            if save_best and is_best:
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(best_path, epoch, model, optimizer, scaler, best_acc, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN + TCN classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint output directory. Default: checkpoints/<config_name>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("training", {})
    ckpt_dir_in_cfg = cfg["training"].get("ckpt_dir")
    ckpt_dir_override: Path | None = args.output_dir
    if ckpt_dir_override is not None:
        ckpt_dir = ckpt_dir_override
    elif ckpt_dir_in_cfg in (None, "", "checkpoints"):
        # 默认放在 checkpoints/<配置文件名> 下
        ckpt_dir = Path("checkpoints") / args.config.stem
    else:
        ckpt_dir = Path(ckpt_dir_in_cfg)
    cfg["training"]["ckpt_dir"] = str(ckpt_dir)
    train_loop(cfg)


if __name__ == "__main__":
    main()
