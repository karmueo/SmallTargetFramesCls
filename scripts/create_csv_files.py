import argparse
import glob
import random
import re
from pathlib import Path
from typing import List, Tuple

# 默认参数，可通过命令行覆盖
DEFAULT_ROOT = Path("./dataset/110_video_frames_2025_11_all_200_have_json").resolve()
DEFAULT_SEED = 0
DEFAULT_RATIOS = (0.9, 0.1, 0.0)  # train, val, test
DEFAULT_SEPARATOR = " "  # 与 kinetics 解析的 PATH_LABEL_SEPARATOR 对齐


def count_frames(dir_path: Path) -> int:
    """统计一个样本目录中的帧数（支持常见图片扩展名）。"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    n = 0
    for e in exts:
        n += len(glob.glob(str(dir_path / e)))
    return n


def load_class_map(path: Path) -> dict:
    """
    从文本加载类别映射，格式：`<id> <folder>` 或 `<id>,<folder>`，支持 # 注释。
    返回 {folder: id}
    """
    mapping = {}
    with open(path) as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            parts = [p for p in re.split(r"[,\s]+", ln) if p]
            if len(parts) != 2:
                raise SystemExit(f"映射行格式错误: {ln}")
            idx, name = parts
            mapping[name] = int(idx)
    if not mapping:
        raise SystemExit(f"映射文件为空: {path}")
    return mapping


def build_class_index(class_dirs: List[Path], override: dict | None = None):
    """
    将类别目录名映射为整数标签。
    - 如果提供 override（{folder: id}），按该映射；缺失或多余目录会报错。
    - 否则按名称排序生成连续整数。
    """
    dir_names = {p.name for p in class_dirs}
    if override:
        missing = set(override.keys()) - dir_names
        extra = dir_names - set(override.keys())
        if missing:
            raise SystemExit(f"映射中存在数据集没有的类别目录: {sorted(missing)}")
        if extra:
            raise SystemExit(f"数据集中存在映射未覆盖的类别目录: {sorted(extra)}")
        return override
    class_names = sorted(dir_names)
    return {name: idx for idx, name in enumerate(class_names)}


def collect_samples(root: Path, class_map: dict | None = None) -> Tuple[List[Tuple[str, int, int]], dict]:
    """收集所有样本，返回 (relative_dir, num_frames, int_label) 列表。"""
    # 仅取一级子目录作为类别
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise SystemExit(f"未找到类别目录，请检查路径: {root}")

    cls2idx = build_class_index(class_dirs, class_map)
    samples = []
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        cls_idx = cls2idx[cls_name]
        # 样本目录为类别目录下的子目录，忽略 .json 文件等非目录文件
        for v in cls_dir.iterdir():
            if v.is_dir():
                n = count_frames(v)
                if n > 0:
                    rel = v.relative_to(root).as_posix()
                    # 写入整数标签，避免字符串标签被误解析
                    samples.append((rel, n, cls_idx))
    if not samples:
        raise SystemExit("未找到任何包含帧的样本目录，请检查数据组织。")
    return samples, cls2idx


def write_txt(path: Path, rows: List[Tuple[str, int, int]], sep: str):
    """写三列文本：frame_dir num_frames label(int)，用指定分隔符，不写表头。"""
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{r[0]}{sep}{r[1]}{sep}{r[2]}\n")


def parse_args():
    ap = argparse.ArgumentParser("Generate train/val/test csv for frame directory dataset")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root dir containing class subdirs")
    ap.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0], help="Train split ratio")
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1], help="Validation split ratio")
    ap.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2], help="Test split ratio; if 0, test=val copy")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--separator", type=str, default=DEFAULT_SEPARATOR, help="Field separator (must match PATH_LABEL_SEPARATOR)")
    ap.add_argument("--output-dir", type=Path, default=None, help="Where to write csv files (default: root)")
    ap.add_argument(
        "--class-map",
        type=Path,
        default="configs/class_map.txt",
        help="Optional mapping file: '<id> <folder>' per line",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.output_dir.resolve() if args.output_dir else root
    if not root.exists():
        raise SystemExit(f"数据根目录不存在: {root}")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    s = sum(ratios)
    if s <= 0:
        raise SystemExit("所有划分比例之和必须 > 0")
    # 归一化，允许用户不严格和为1
    ratios = tuple(r / s for r in ratios)
    random.seed(args.seed)
    class_map = load_class_map(args.class_map) if args.class_map else None
    samples, cls2idx = collect_samples(root, class_map)
    random.shuffle(samples)
    n_total = len(samples)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    # 剩余为 test
    # 剩余样本作为测试集（无需单独变量）
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]
    # 若用户 test_ratio 为 0 则用 val 作为 test 备份
    if args.test_ratio == 0.0:
        test = val
    out_dir.mkdir(parents=True, exist_ok=True)
    write_txt(out_dir / "train.csv", train, args.separator)
    write_txt(out_dir / "val.csv", val, args.separator)
    write_txt(out_dir / "test.csv", test, args.separator)
    print(f"写入: {out_dir / 'train.csv'} ({len(train)})")
    print(f"写入: {out_dir / 'val.csv'} ({len(val)})")
    print(f"写入: {out_dir / 'test.csv'} ({len(test)})")
    print("类别映射：")
    for name, idx in sorted(cls2idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
