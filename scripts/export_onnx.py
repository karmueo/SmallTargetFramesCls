from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
import torch
import torch.onnx
from torch.nn.utils import parametrize

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - 运行时检查依赖
    raise SystemExit("需要安装 onnxruntime 才能导出后对比推理：pip install onnxruntime") from exc

from scripts import eval as eval_module
from scripts import train as train_module


class OnnxLogitsWrapper(torch.nn.Module):
    """封装模型，只导出 logits，避免字典输出影响 ONNX。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, boxes: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.model(images, boxes=boxes)
        return outputs["logits"]


def export_to_onnx(
    model: torch.nn.Module,
    images: torch.Tensor,
    boxes: torch.Tensor | None,
    output_path: Path,
    opset: int,
) -> None:
    """导出模型到 ONNX 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = OnnxLogitsWrapper(model)
    # torch.onnx.export 过程中可能会改变 module.training（已在本仓库模型上复现会被切回 train），
    # 这会让后续对比阶段的 Dropout/BN 行为变化，导致“误差很大”的假象。这里显式保护并恢复状态。
    was_training = model.training
    wrapper.eval()
    dynamic_axes = {
        "images": {0: "batch", 1: "time"},
        "logits": {0: "batch"},
    }
    input_names = ["images"]
    args = (images, )
    if boxes is not None:
        dynamic_axes["boxes"] = {0: "batch", 1: "time"}
        input_names.append("boxes")
        args = (images, boxes)
    torch.onnx.export(
        wrapper,
        args=args,
        f=str(output_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    # 恢复导出前训练/评估模式，避免影响后续 PyTorch 推理
    model.train(was_training)


def create_ort_session(onnx_path: Path) -> ort.InferenceSession:
    """创建 ONNX Runtime Session。"""
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def strip_weight_norms(module: torch.nn.Module) -> int:
    """
    移除模型中的 weight_norm（包含 parametrization 形式），避免导出时权重还原不一致。
    返回移除的模块数量，若仍有未移除则抛出异常。
    """
    removed = 0
    for name, sub in module.named_modules():
        if parametrize.is_parametrized(sub, "weight"):
            try:
                parametrize.remove_parametrizations(sub, "weight", leave_parametrized=False)
                removed += 1
                continue
            except Exception as exc:  # pragma: no cover - 导出时检查
                raise RuntimeError(f"移除 weight_norm 参数化失败: {name}: {exc}") from exc
        try:
            torch.nn.utils.remove_weight_norm(sub)
            removed += 1
        except ValueError:
            continue
        except Exception as exc:  # pragma: no cover - 导出时检查
            raise RuntimeError(f"移除 weight_norm 失败: {name}: {exc}") from exc

    remaining = 0
    for _, sub in module.named_modules():
        if parametrize.is_parametrized(sub, "weight"):
            remaining += 1
    if remaining > 0:
        raise RuntimeError(f"仍有 {remaining} 处 weight_norm 未移除")
    return removed


@torch.no_grad()
def compare_with_onnx(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> Dict[str, float]:
    """对比 PyTorch 与 ONNX 推理输出误差。"""
    model.eval()
    total_abs = 0.0
    total_elems = 0
    max_abs = 0.0
    mismatch = 0
    prob_mean_abs = 0.0
    prob_max_abs = 0.0
    mean_shift_abs = 0.0
    centered_total_abs = 0.0
    centered_max_abs = 0.0
    samples = 0
    batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        boxes = batch.get("boxes")
        if boxes is not None:
            boxes = boxes.to(device, non_blocking=True)

        pt_logits = model(images, boxes=boxes)["logits"].cpu()
        ort_inputs = {"images": images.cpu().numpy()}
        if boxes is not None:
            ort_inputs["boxes"] = boxes.cpu().numpy()
        ort_logits = torch.from_numpy(session.run(None, ort_inputs)[0])

        diff = torch.abs(pt_logits - ort_logits)
        total_abs += diff.sum().item()
        total_elems += diff.numel()
        max_abs = max(max_abs, diff.max().item())
        mismatch += (pt_logits.argmax(dim=1) != ort_logits.argmax(dim=1)).sum().item()

        # 评估去均值后的差异，检查是否存在整体偏移
        pt_mean = pt_logits.mean(dim=1, keepdim=True)
        ort_mean = ort_logits.mean(dim=1, keepdim=True)
        mean_shift_abs += torch.abs(pt_mean - ort_mean).sum().item()
        centered_diff = torch.abs((pt_logits - pt_mean) - (ort_logits - ort_mean))
        centered_total_abs += centered_diff.sum().item()
        centered_max_abs = max(centered_max_abs, centered_diff.max().item())

        pt_prob = pt_logits.softmax(dim=1)
        ort_prob = ort_logits.softmax(dim=1)
        prob_diff = torch.abs(pt_prob - ort_prob)
        prob_mean_abs += prob_diff.sum().item()
        prob_max_abs = max(prob_max_abs, prob_diff.max().item())

        samples += pt_logits.size(0)
        batches += 1

    return {
        "mean_abs": total_abs / max(total_elems, 1),
        "max_abs": max_abs,
        "mismatch_rate": mismatch / max(samples, 1),
        "prob_mean_abs": prob_mean_abs / max(total_elems, 1),
        "prob_max_abs": prob_max_abs,
        "mean_shift_abs": mean_shift_abs / max(samples, 1),
        "centered_mean_abs": centered_total_abs / max(total_elems, 1),
        "centered_max_abs": centered_max_abs,
        "samples": float(samples),
        "batches": float(batches),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出模型为 ONNX 并对比 PyTorch 推理误差")
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18.yaml"), help="配置文件路径")
    parser.add_argument("--checkpoint", type=Path, required=True, help=".pt 权重路径")
    parser.add_argument("--onnx-path", type=Path, default=None, help="导出 ONNX 文件路径，默认与 checkpoint 同名")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--root", type=Path, default=None, help="可选：覆盖数据根目录")
    parser.add_argument("--test-csv", type=Path, default=None, help="可选：覆盖测试 CSV")
    parser.add_argument("--num-workers", type=int, default=None, help="可选：覆盖 DataLoader num_workers")
    parser.add_argument("--pin-memory", type=int, default=None, help="可选：覆盖 DataLoader pin_memory（0/1）")
    parser.add_argument(
        "--strip-weight-norm",
        action="store_true",
        help="可选：导出前移除 weight_norm（默认不移除，避免改变模型结构）",
    )
    parser.add_argument("--device", type=str, default="cpu", help='推理使用的设备，默认 "cpu"')
    parser.add_argument("--max-batches", type=int, default=5, help="最多对比的 batch 数量，0 表示全量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = train_module.load_config(args.config)
    if args.num_workers is not None:
        cfg.setdefault("data", {})["num_workers"] = int(args.num_workers)
    if args.pin_memory is not None:
        cfg.setdefault("data", {})["pin_memory"] = bool(int(args.pin_memory))
    device = torch.device(args.device)

    loader, class_to_idx = eval_module.build_test_loader(cfg, root_override=args.root, test_csv_override=args.test_csv)
    model = eval_module.load_model(cfg, class_to_idx=class_to_idx, checkpoint=args.checkpoint, device=device)
    model.float()
    if args.strip_weight_norm:
        removed = strip_weight_norms(model)
        if removed > 0:
            print(f"已移除 {removed} 处 weight_norm 参数化/钩子")

    # 使用首个 batch 的真实形状进行导出，保证时序长度等与数据一致
    try:
        first_batch = next(iter(loader))
    except StopIteration as exc:
        raise SystemExit("测试集为空，无法导出示例输入") from exc
    example_images = first_batch["images"].to(device)
    example_boxes = first_batch.get("boxes")
    if example_boxes is not None:
        example_boxes = example_boxes.to(device)

    onnx_path = args.onnx_path or args.checkpoint.with_suffix(".onnx")
    export_to_onnx(model, example_images, example_boxes, onnx_path, opset=args.opset)
    print(f"ONNX 已导出到 {onnx_path}")

    session = create_ort_session(onnx_path)
    max_batches = None if args.max_batches == 0 else args.max_batches
    metrics = compare_with_onnx(model, session, loader, device, max_batches=max_batches)
    print(
        "对比完成："
        f" mean_abs={metrics['mean_abs']:.6f},"
        f" max_abs={metrics['max_abs']:.6f},"
        f" mismatch_rate={metrics['mismatch_rate']*100:.2f}%"
        f" prob_mean_abs={metrics['prob_mean_abs']:.6f},"
        f" prob_max_abs={metrics['prob_max_abs']:.6f}"
        f" mean_shift_abs={metrics['mean_shift_abs']:.6f},"
        f" centered_mean_abs={metrics['centered_mean_abs']:.6f},"
        f" centered_max_abs={metrics['centered_max_abs']:.6f}"
        f" over {int(metrics['samples'])} samples"
        f" ({int(metrics['batches'])} batches)"
    )


if __name__ == "__main__":
    main()
