from pathlib import Path

import numpy as np
import torch
import torchvision
from datasets import Dataset


def dict_to_dataset(data_dict):
    """
    (N, d) tensor dict -> HuggingFace Dataset
    rows are returned as torch.Tensor after set_format
    """
    processed = {k: v.tolist() for k, v in data_dict.items()}
    dataset = Dataset.from_dict(processed)
    dataset.set_format(type="torch", columns=list(data_dict.keys()))
    return dataset


SCRIPT_DIR = Path(__file__).resolve().parent


def ensure_cifar10_dataset(download_root):
    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/loading CIFAR-10 at: {download_root}")
    train_ds = torchvision.datasets.CIFAR10(
        root=str(download_root),
        train=True,
        download=True,
    )
    valid_ds = torchvision.datasets.CIFAR10(
        root=str(download_root),
        train=False,
        download=True,
    )
    return train_ds, valid_ds


def save_cifar10_dataset(output_root=None, download_root=None):
    if output_root is None:
        output_root = SCRIPT_DIR / "parsed"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = SCRIPT_DIR / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if download_root is None:
        download_root = output_root / "raw"
    else:
        download_root = Path(download_root)
        if not download_root.is_absolute():
            download_root = SCRIPT_DIR / download_root

    train_ds, valid_ds = ensure_cifar10_dataset(download_root)

    print("Converting CIFAR-10 arrays...")
    # torchvision CIFAR10 stores images as uint8 arrays with shape (N, 32, 32, 3)
    # Build train set with horizontal flip augmentation:
    # 50,000 original + 50,000 flipped = 100,000.
    x_train_orig = np.asarray(train_ds.data)
    y_train_orig = np.asarray(train_ds.targets, dtype=np.int64)

    x_train_flip = np.flip(x_train_orig, axis=2).copy()
    y_train_flip = y_train_orig.copy()

    x_train = np.concatenate([x_train_orig, x_train_flip], axis=0).reshape(-1, 32 * 32 * 3)
    y_train = np.concatenate([y_train_orig, y_train_flip], axis=0)
    if x_train.shape[0] != 100000:
        raise ValueError(f"Expected 100000 train samples after hflip augmentation, got {x_train.shape[0]}")

    x_valid = np.asarray(valid_ds.data).reshape(len(valid_ds), -1)
    y_valid = np.asarray(valid_ds.targets, dtype=np.int64)

    train_data = {
        "x_clean": torch.tensor(x_train, dtype=torch.uint8),
        "label": torch.tensor(y_train, dtype=torch.int64),
    }
    valid_data = {
        "x_clean": torch.tensor(x_valid, dtype=torch.uint8),
        "label": torch.tensor(y_valid, dtype=torch.int64),
    }

    train_data_path = output_root / "train.pt"
    valid_data_path = output_root / "valid.pt"
    torch.save(train_data, train_data_path)
    torch.save(valid_data, valid_data_path)
    print(f"Train sample count (with hflip): {train_data['x_clean'].shape[0]}")
    print(f"Valid sample count: {valid_data['x_clean'].shape[0]}")
    print(f"Saved raw train tensor dict: {train_data_path}")
    print(f"Saved raw valid tensor dict: {valid_data_path}")

    train_dataset = dict_to_dataset(train_data)
    valid_dataset = dict_to_dataset(valid_data)

    train_dataset_path = output_root / "base" / "train"
    valid_dataset_path = output_root / "base" / "valid"
    train_dataset.save_to_disk(str(train_dataset_path))
    valid_dataset.save_to_disk(str(valid_dataset_path))
    print(f"Saved HF train dataset: {train_dataset_path}")
    print(f"Saved HF valid dataset: {valid_dataset_path}")

    return {
        "download_root": str(download_root),
        "train_data_path": str(train_data_path),
        "valid_data_path": str(valid_data_path),
        "train_dataset_path": str(train_dataset_path),
        "valid_dataset_path": str(valid_dataset_path),
    }


if __name__ == "__main__":
    paths = save_cifar10_dataset(
        output_root=SCRIPT_DIR / "parsed",
    )
    print(paths)
