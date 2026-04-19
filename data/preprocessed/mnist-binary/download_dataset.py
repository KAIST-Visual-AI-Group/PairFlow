from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
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

BASE_URL = (
    "https://raw.githubusercontent.com/aiddun/binary-mnist/master/"
    "original_28x28/all_digits_binary_pixels"
)
FILES = ("x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy")


def download_file(url, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        print(f"Already exists, skipping: {dst_path}")
        return
    print(f"Downloading {url} -> {dst_path}")
    urlretrieve(url, str(dst_path))


def ensure_dataset_files(download_root):
    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    local_paths = {}
    for file_name in FILES:
        url = f"{BASE_URL}/{file_name}"
        dst = download_root / file_name
        download_file(url, dst)
        local_paths[file_name] = dst
    return local_paths


def save_mnist_binary_dataset(output_root=None, download_root=None):
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

    local_paths = ensure_dataset_files(download_root)

    print("Loading numpy arrays...")
    x_train = np.load(local_paths["x_train.npy"])
    y_train = np.load(local_paths["y_train.npy"])
    x_test = np.load(local_paths["x_test.npy"])
    y_test = np.load(local_paths["y_test.npy"])

    # Expected shape: (N, 28, 28) -> flatten to (N, 784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Binary pixels in {0,1}; cast to int8 for storage compatibility.
    train_data = {
        "x_clean": torch.tensor(x_train, dtype=torch.int8),
        "label": torch.tensor(y_train, dtype=torch.int64),
    }
    valid_data = {
        "x_clean": torch.tensor(x_test, dtype=torch.int8),
        "label": torch.tensor(y_test, dtype=torch.int64),
    }

    train_data_path = output_root / "train.pt"
    valid_data_path = output_root / "valid.pt"
    torch.save(train_data, train_data_path)
    torch.save(valid_data, valid_data_path)
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
    paths = save_mnist_binary_dataset(
        output_root=SCRIPT_DIR / "parsed",
    )
    print(paths)
