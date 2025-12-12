from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameSamplingConfig:
    """帧采样配置。"""

    target_length: int
    strategy: str = "uniform"  # uniform | random_offset | fixed_rate
    sampling_rate: int = 1  # 仅在 fixed_rate 下使用，表示步长
    pad_mode: str = "repeat"  # repeat | loop | mirror
    jitter: bool = False  # 仅在 random_offset 时有效
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.target_length <= 0:
            raise ValueError("target_length must be > 0")
        if self.strategy not in {"uniform", "random_offset", "fixed_rate"}:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        if self.strategy == "fixed_rate" and self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be > 0 for fixed_rate strategy")
        if self.pad_mode not in {"repeat", "loop", "mirror"}:
            raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")


@dataclass
class SequenceRecord:
    """描述单个序列的元数据。"""

    seq_dir: Path
    meta_path: Path
    label: str
    class_index: int
    resolution: Tuple[int, int]
    frames: List[Path]
    boxes: List[Tuple[float, float, float, float]]  # 归一化后的 (cx, cy, w, h)


def _clamp_bbox(
    bbox: Dict[str, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """将 bbox 限制在图像内，并返回中心点与宽高（归一化前）。"""
    x1 = max(0.0, float(bbox["x"]))
    y1 = max(0.0, float(bbox["y"]))
    x2 = min(float(width), x1 + float(bbox["w"]))
    y2 = min(float(height), y1 + float(bbox["h"]))
    clamped_w = max(1.0, x2 - x1)
    clamped_h = max(1.0, y2 - y1)
    cx = x1 + clamped_w / 2.0
    cy = y1 + clamped_h / 2.0
    return cx, cy, clamped_w, clamped_h


def sample_frame_indices(
    num_frames: int,
    sampling: FrameSamplingConfig,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    根据配置采样帧索引，若序列长度不足则按 pad_mode 填充。
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    rng = rng or random.Random(sampling.seed)

    if sampling.strategy == "fixed_rate":
        # 固定步长采样，超出范围时截断到最后一帧，始终返回 target_length 个索引
        return [
            min(i * sampling.sampling_rate, num_frames - 1)
            for i in range(sampling.target_length)
        ]

    if num_frames >= sampling.target_length:
        stride = (num_frames - 1) / max(sampling.target_length - 1, 1)
        indices: List[int] = []
        for i in range(sampling.target_length):
            base = int(round(i * stride))
            if sampling.strategy == "random_offset" and sampling.jitter:
                window = max(int(stride), 1)
                low = max(0, base - window)
                high = min(num_frames - 1, base + window)
                base = rng.randint(low, high)
            indices.append(min(base, num_frames - 1))
        return indices

    indices = list(range(num_frames))
    if sampling.pad_mode == "repeat":
        while len(indices) < sampling.target_length:
            indices.append(indices[-1])
    elif sampling.pad_mode == "loop":
        ptr = 0
        while len(indices) < sampling.target_length:
            indices.append(indices[ptr % num_frames])
            ptr += 1
    elif sampling.pad_mode == "mirror":
        forward = True
        ptr = num_frames - 2 if num_frames > 1 else 0
        while len(indices) < sampling.target_length:
            indices.append(ptr)
            if num_frames == 1:
                continue
            if forward:
                ptr -= 1
                if ptr < 0:
                    ptr = 1
                    forward = False
            else:
                ptr += 1
                if ptr >= num_frames:
                    ptr = num_frames - 2
                    forward = True
    return indices


class SequenceDataset(Dataset):
    """
    从 CSV 列表加载序列数据，完成帧采样与基础一致性校验。

    CSV 格式：
        seq_path,label
        bird/seq_0001,bird
        drone/seq_0002,drone
    seq_path 为相对 root_dir 的目录路径，同名 JSON 位于其父目录。
    """

    def __init__(
        self,
        csv_path: Path | str,
        root_dir: Optional[Path | str],
        sampling: FrameSamplingConfig,
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        return_tensor: bool = True,
        skip_broken: bool = True,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.csv_path = Path(csv_path)
        self.sampling = sampling
        self.image_transform = image_transform
        self.return_tensor = return_tensor
        self.skip_broken = skip_broken
        self._rng = random.Random(sampling.seed)

        records = self._read_csv(self.csv_path)
        classes = sorted({label for _, label in records})
        if class_to_idx is None:
            self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        self.samples: List[SequenceRecord] = self._build_samples(records)

    def _read_csv(self, csv_path: Path) -> List[Tuple[Path, str]]:
        """
        支持两种格式：
        1) 逗号分隔：seq_path,label
        2) 空白分隔：seq_path num_frames label_id   （label 字段缺失时从路径父目录推断）
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        items: List[Tuple[Path, str]] = []
        with csv_path.open("r", newline="") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts: List[str]
                if "," in line:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                else:
                    parts = re.split(r"\s+", line)
                if not parts:
                    continue
                seq_rel = parts[0]
                label: Optional[str] = None
                for token in parts[1:]:
                    if not token.isdigit():
                        label = token
                        break
                if label is None:
                    seq_path = Path(seq_rel)
                    # 默认以一级父目录名作为类别
                    if len(seq_path.parts) >= 2:
                        label = seq_path.parts[0]
                    else:
                        label = seq_path.name
                seq_dir = Path(seq_rel)
                if self.root_dir is not None:
                    seq_dir = self.root_dir / seq_dir
                items.append((seq_dir, label))
        if not items:
            raise ValueError(f"No samples found in {csv_path}")
        return items

    def _build_samples(self, rows: Iterable[Tuple[Path, str]]) -> List[SequenceRecord]:
        samples: List[SequenceRecord] = []
        for seq_dir, label in rows:
            meta_path = seq_dir.parent / f"{seq_dir.name}.json"
            try:
                record = self._load_sequence(seq_dir, meta_path, label)
                samples.append(record)
            except Exception as exc:
                if self.skip_broken:
                    logger.warning("Skip broken sample %s: %s", seq_dir, exc)
                    continue
                raise
        if not samples:
            raise ValueError("No valid samples after validation")
        return samples

    def _load_sequence(self, seq_dir: Path, meta_path: Path, label: str) -> SequenceRecord:
        if not seq_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory missing: {seq_dir}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Metadata file missing: {meta_path}")

        with meta_path.open("r") as f:
            meta = json.load(f)
        width = int(meta.get("resolution", {}).get("width", 0))
        height = int(meta.get("resolution", {}).get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution in {meta_path}")

        tracks = meta.get("tracks", [])
        if not tracks:
            raise ValueError(f"No tracks found in {meta_path}")

        frames: List[Path] = []
        boxes: List[Tuple[float, float, float, float]] = []
        for track in tracks:
            image_name = track.get("image")
            bbox = track.get("bbox")
            if image_name is None or bbox is None:
                raise ValueError(f"Track missing fields in {meta_path}")
            img_path = seq_dir / image_name
            if not img_path.is_file():
                raise FileNotFoundError(f"Frame not found: {img_path}")
            cx, cy, bw, bh = _clamp_bbox(bbox, width, height)
            frames.append(img_path)
            boxes.append((cx / width, cy / height, bw / width, bh / height))

        class_index = self.class_to_idx.get(label)
        if class_index is None:
            raise ValueError(f"Label {label} not found in class_to_idx mapping")
        return SequenceRecord(
            seq_dir=seq_dir,
            meta_path=meta_path,
            label=label,
            class_index=class_index,
            resolution=(width, height),
            frames=frames,
            boxes=boxes,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.samples[index]
        indices = sample_frame_indices(len(record.frames), self.sampling, rng=self._rng)
        images: List[torch.Tensor | Image.Image] = []
        boxes: List[Tuple[float, float, float, float]] = []
        for idx in indices:
            img = Image.open(record.frames[idx]).convert("RGB")
            if self.image_transform is not None:
                img = self.image_transform(img)
            elif self.return_tensor:
                img = torch.from_numpy(np.array(img))
            images.append(img)
            boxes.append(record.boxes[idx])

        sample: Dict[str, object] = {
            "images": torch.stack(images) if self.return_tensor and isinstance(images[0], torch.Tensor) else images,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "label": record.class_index,
            "label_name": record.label,
            "frame_indices": indices,
            "resolution": record.resolution,
            "seq_path": str(record.seq_dir),
        }
        return sample
