#!/usr/bin/env python3
"""
随机分割文件夹中的子文件夹为两部分。
指定一个数量，将文件夹下的子文件夹分成两份，并保存到新文件夹下。
子文件夹内的文件结构不会修改。
"""

import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple


def get_all_subfolders(folder_path: str) -> List[str]:
    """
    获取指定文件夹下的所有直接子文件夹（不包括嵌套子文件夹）。
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        子文件夹列表，每个元素为相对于文件夹的相对路径
    """
    subfolders = []
    folder_path_obj = Path(folder_path)
    
    # 遍历直接子文件夹
    for item in folder_path_obj.iterdir():
        if item.is_dir():
            # 保存相对路径
            relative_path = item.relative_to(folder_path_obj)
            subfolders.append(str(relative_path))
    
    return sorted(subfolders)  # 排序以确保一致性


def split_files(
    source_folder: str,
    output_folder: str,
    count: int,
    random_seed=None
) -> Tuple[List[str], List[str]]:
    """
    随机分割子文件夹为两部分。
    
    Args:
        source_folder: 源文件夹路径
        output_folder: 输出文件夹路径
        count: 第一部分的子文件夹数量
        random_seed: 随机种子（可选）
        
    Returns:
        元组 (first_part, second_part)，每个都是子文件夹列表
    """
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    
    # 验证源文件夹存在
    if not source_path.exists():
        raise FileNotFoundError(f"源文件夹不存在: {source_folder}")
    
    # 获取所有子文件夹
    all_subfolders = get_all_subfolders(source_folder)
    
    if not all_subfolders:
        raise ValueError(f"源文件夹中没有子文件夹: {source_folder}")
    
    total_subfolders = len(all_subfolders)
    
    # 验证指定的数量
    if count < 0 or count > total_subfolders:
        raise ValueError(
            f"指定的数量({count})无效。子文件夹总数为{total_subfolders}"
        )
    
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
    
    # 随机打乱子文件夹列表
    shuffled_subfolders = all_subfolders.copy()
    random.shuffle(shuffled_subfolders)
    
    # 分割为两部分
    first_part = shuffled_subfolders[:count]
    second_part = shuffled_subfolders[count:]
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建两个子文件夹
    first_folder = output_path / "part_1"
    second_folder = output_path / "part_2"
    first_folder.mkdir(exist_ok=True)
    second_folder.mkdir(exist_ok=True)
    
    # 复制子文件夹到第一部分
    print(f"正在复制第一部分子文件夹 ({len(first_part)} 个)...")
    for subfolder_rel_path in first_part:
        src_subfolder = source_path / subfolder_rel_path
        dst_subfolder = first_folder / subfolder_rel_path
        
        # 递归复制整个子文件夹及其内容
        shutil.copytree(src_subfolder, dst_subfolder)
        
        # 同步复制同名的 .json 文件
        src_json = source_path / f"{subfolder_rel_path}.json"
        if src_json.exists():
            dst_json = first_folder / f"{subfolder_rel_path}.json"
            shutil.copy2(src_json, dst_json)
    
    # 复制子文件夹到第二部分
    print(f"正在复制第二部分子文件夹 ({len(second_part)} 个)...")
    for subfolder_rel_path in second_part:
        src_subfolder = source_path / subfolder_rel_path
        dst_subfolder = second_folder / subfolder_rel_path
        
        # 递归复制整个子文件夹及其内容
        shutil.copytree(src_subfolder, dst_subfolder)
        
        # 同步复制同名的 .json 文件
        src_json = source_path / f"{subfolder_rel_path}.json"
        if src_json.exists():
            dst_json = second_folder / f"{subfolder_rel_path}.json"
            shutil.copy2(src_json, dst_json)
    
    print("\n文件夹分割完成！")
    print(f"第一部分: {len(first_part)} 个子文件夹 -> {first_folder}")
    print(f"第二部分: {len(second_part)} 个子文件夹 -> {second_folder}")
    
    if random_seed is not None:
        print(f"使用的随机种子: {random_seed}")
    
    return first_part, second_part


def main():
    parser = argparse.ArgumentParser(
        description="随机分割文件夹中的子文件夹为两部分"
    )
    parser.add_argument(
        "source",
        type=str,
        help="源文件夹路径（包含要分割的子文件夹）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件夹路径（默认为源文件夹同级目录）"
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="第一部分的子文件夹数量"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选，不指定则每次随机）"
    )
    
    args = parser.parse_args()
    
    # 处理输出文件夹路径
    if args.output is None:
        source_path = Path(args.source)
        args.output = str(source_path.parent / f"{source_path.name}_split")
    
    try:
        split_files(
            source_folder=args.source,
            output_folder=args.output,
            count=args.count,
            random_seed=args.seed
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == "__main__":
    main()
