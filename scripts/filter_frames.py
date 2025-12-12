import argparse
import glob
import shutil
from pathlib import Path
from typing import Tuple

# 默认参数
DEFAULT_ROOT = Path("dataset/110_video_frames_2025_11_all_200_have_json").resolve()
DEFAULT_MIN_FRAMES = 100


def count_frames(dir_path: Path) -> int:
    """统计一个样本目录中的帧数（支持常见图片扩展名）。"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    n = 0
    for e in exts:
        n += len(glob.glob(str(dir_path / e)))
    return n


def filter_samples(root: Path, min_frames: int) -> Tuple[int, int]:
    """
    遍历所有类别和样本，删除帧数小于 min_frames 的视频文件夹及对应的 .json 文件。
    返回 (总删除数, 保留数)
    """
    # 仅取一级子目录作为类别
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise SystemExit(f"未找到类别目录，请检查路径: {root}")

    deleted_count = 0
    kept_count = 0
    
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        print(f"\n处理类别: {cls_name}")
        
        # 样本目录为类别目录下的子目录
        for v in cls_dir.iterdir():
            if v.is_dir():
                n = count_frames(v)
                if n < min_frames:
                    print(f"  删除: {v.name} (帧数: {n} < {min_frames})")
                    shutil.rmtree(v)
                    # 删除对应的 .json 文件
                    json_file = cls_dir / f"{v.name}.json"
                    if json_file.exists():
                        json_file.unlink()
                        print(f"    同步删除: {json_file.name}")
                    deleted_count += 1
                elif n > 0:
                    kept_count += 1


    return deleted_count, kept_count


def parse_args():
    ap = argparse.ArgumentParser("Filter video frame folders by minimum frame count")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root dir containing class subdirs")
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES, help="Minimum frame count to keep")
    ap.add_argument("--dry-run", action="store_true", help="只显示将被删除的文件夹，不实际删除")
    return ap.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    
    if not root.exists():
        raise SystemExit(f"数据根目录不存在: {root}")
    
    print(f"数据根目录: {root}")
    print(f"最少帧数: {args.min_frames}")
    
    if args.dry_run:
        print("模式: 干运行（仅显示，不删除）\n")
        
        class_dirs = [p for p in root.iterdir() if p.is_dir()]
        total_delete = 0
        total_keep = 0
        
        for cls_dir in class_dirs:
            cls_name = cls_dir.name
            print(f"类别: {cls_name}")
            
            for v in cls_dir.iterdir():
                if v.is_dir():
                    n = count_frames(v)
                    if n < args.min_frames:
                        json_file = cls_dir / f"{v.name}.json"
                        json_status = f" + {json_file.name}" if json_file.exists() else ""
                        print(f"  [删除] {v.name} (帧数: {n}){json_status}")
                        total_delete += 1
                    elif n > 0:
                        print(f"  [保留] {v.name} (帧数: {n})")
                        total_keep += 1
            print()
        
        print(f"总计: 将删除 {total_delete} 个文件夹，保留 {total_keep} 个文件夹")
    else:
        print("模式: 实际删除\n")
        deleted, kept = filter_samples(root, args.min_frames)
        print("\n完成!")
        print(f"已删除: {deleted} 个文件夹")
        print(f"已保留: {kept} 个文件夹")


if __name__ == "__main__":
    main()
