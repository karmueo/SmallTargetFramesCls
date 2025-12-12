from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.data import FrameSamplingConfig, SequenceDataset, sample_frame_indices


DATA_ROOT = Path("dataset/110_video_frames_2025_11_all_200_have_json")


def test_whitespace_csv_and_basic_loading():
    csv_path = DATA_ROOT / "val.csv"
    sampling = FrameSamplingConfig(target_length=8, strategy="uniform", pad_mode="repeat", seed=42)
    dataset = SequenceDataset(
        csv_path=csv_path,
        root_dir=DATA_ROOT,
        sampling=sampling,
        image_transform=None,
        return_tensor=True,
        skip_broken=False,
    )

    assert len(dataset) > 0
    sample = dataset[0]

    images = sample["images"]
    boxes = sample["boxes"]

    assert isinstance(images, torch.Tensor)
    assert images.shape[0] == sampling.target_length
    assert boxes.shape == (sampling.target_length, 4)
    assert (boxes >= 0).all()
    assert (boxes <= 1).all()

    assert isinstance(sample["label"], int)
    assert isinstance(sample["label_name"], str)
    assert sample["label_name"]  # 非空

    width, height = sample["resolution"]
    assert width > 0 and height > 0
    assert len(sample["frame_indices"]) == sampling.target_length


def test_label_inferred_from_path_parent():
    csv_path = DATA_ROOT / "train.csv"
    sampling = FrameSamplingConfig(target_length=4, strategy="uniform", pad_mode="repeat", seed=7)
    dataset = SequenceDataset(
        csv_path=csv_path,
        root_dir=DATA_ROOT,
        sampling=sampling,
        image_transform=None,
        return_tensor=True,
        skip_broken=False,
    )

    seq_path = Path(dataset.samples[0].seq_dir)
    expected_label = seq_path.parts[-2] if len(seq_path.parts) >= 2 else seq_path.name
    assert dataset.samples[0].label == expected_label
    assert dataset.samples[0].class_index in dataset.class_to_idx.values()


def test_fixed_rate_sampling_indices():
    sampling = FrameSamplingConfig(target_length=4, strategy="fixed_rate", sampling_rate=3)
    indices = sample_frame_indices(num_frames=5, sampling=sampling)
    # 0,3,4,4 (最后超界截断到末尾)
    assert indices == [0, 3, 4, 4]
