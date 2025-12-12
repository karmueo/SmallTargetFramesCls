from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train as train_module


def human_readable(num: float) -> str:
    """将数字转为可读的单位表示。"""
    units = ["", "K", "M", "B"]
    for unit in units:
        if abs(num) < 1000.0:
            return f"{num:.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}T"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _conv2d_flops(module: nn.Conv2d, output: torch.Tensor) -> int:
    batch, out_c, out_h, out_w = output.shape
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
    return int(batch * out_c * out_h * out_w * kernel_ops * 2)


def _conv1d_flops(module: nn.Conv1d, output: torch.Tensor) -> int:
    batch, out_c, out_l = output.shape
    kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
    return int(batch * out_c * out_l * kernel_ops * 2)


def _linear_flops(module: nn.Linear, output: torch.Tensor) -> int:
    batch = output.shape[0]
    return int(batch * module.in_features * module.out_features * 2)


def count_flops(model: nn.Module, sample: Dict[str, torch.Tensor]) -> int:
    """
    基于一次前向推理的简单 FLOPs 估计（仅统计 Conv1d/Conv2d/Linear）。
    """
    flops = 0
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hook(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            hooks.append(
                module.register_forward_hook(
                    lambda m, _inp, out: nonlocal_add(_conv2d_flops(m, out))
                )
            )
        elif isinstance(module, nn.Conv1d):
            hooks.append(
                module.register_forward_hook(
                    lambda m, _inp, out: nonlocal_add(_conv1d_flops(m, out))
                )
            )
        elif isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(
                    lambda m, _inp, out: nonlocal_add(_linear_flops(m, out))
                )
            )

    def nonlocal_add(value: int) -> None:
        nonlocal flops
        flops += int(value)

    for module in model.modules():
        register_hook(module)

    model.eval()
    with torch.no_grad():
        model(sample["images"], boxes=sample.get("boxes"))

    for h in hooks:
        h.remove()
    return flops


def build_dummy_sample(cfg: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    t = int(data_cfg["frame_sampling"]["target_length"])
    c = int(model_cfg.get("in_channels", 3))
    size = int(data_cfg["image_size"])
    images = torch.randn(1, t, c, size, size, device=device)
    boxes = torch.zeros(1, t, 4, device=device) if model_cfg.get("use_position", False) else None
    return {"images": images, "boxes": boxes}


def main() -> None:
    parser = argparse.ArgumentParser(description="计算模型参数量与 FLOPs（估算）")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="配置文件路径")
    parser.add_argument("--device", type=str, default="cpu", help='设备字符串，如 "cpu" 或 "cuda:0"')
    args = parser.parse_args()

    cfg = train_module.load_config(args.config)
    device = torch.device(args.device)

    class_map_path = Path(cfg["data"]["class_map"])
    class_to_idx = train_module.load_class_map(class_map_path)
    num_classes = len(class_to_idx)

    model = train_module.build_model(cfg, num_classes=num_classes, device=device)
    sample = build_dummy_sample(cfg, device=device)

    total_params = count_parameters(model)
    flops = count_flops(model, sample)

    print(f"参数量: {total_params} ({human_readable(total_params)})")
    print(f"FLOPs(一次前向): {flops} ({human_readable(flops)})")


if __name__ == "__main__":
    main()
