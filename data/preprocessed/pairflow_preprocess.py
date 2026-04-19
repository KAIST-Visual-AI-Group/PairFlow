import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import sys

import datasets
from datasets import Dataset

DATASET_CONFIGS = {
    "qm9": {"D": 32, "V": 40},
    # D = padded length from zinc250k tokenizer (see load_zinc250k.ipynb, max_length=74).
    "zinc-250k": {"D": 74, "V": 72},
    "mnist-binary": {"D": 28*28, "V": 2},
    "cifar-10": {"D": 32*32*3, "V": 256},
}


def resolve_data_root(path_str):
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    path = Path(path_str)
    if not path.is_absolute():
        path = script_dir / path
    return path


def load_train_data(data_path, seq_len, dataset_type=None):
    data_path = resolve_data_root(data_path)
    if data_path.is_dir():
        train_ds = datasets.load_from_disk(str(data_path))
        train_ds.set_format(type="torch")
        input_key = "x_clean" if "x_clean" in train_ds.column_names else "input_ids"
        if input_key not in train_ds.column_names:
            raise KeyError(f"Neither x_clean nor input_ids found in {data_path}")
        # Keep all original columns (labels/components), and normalize main keys.
        data = {col: train_ds[col] for col in train_ds.column_names}
        if dataset_type == "cifar-10":
            # CIFAR-10 raw pixels are in [0, 255], so int8 would overflow.
            data["x_clean"] = data[input_key].to(torch.uint8)
        else:
            data["x_clean"] = data[input_key].to(torch.int8)
        if "attention_mask" in data:
            data["attention_mask"] = data["attention_mask"].to(torch.int8)
    else:
        data = torch.load(str(data_path))
        if "x_clean" not in data and "input_ids" in data:
            data["x_clean"] = data["input_ids"]
        if "x_clean" not in data:
            raise KeyError(f"x_clean (or input_ids) not found in {data_path}")
        if dataset_type == "cifar-10":
            data["x_clean"] = data["x_clean"].to(torch.uint8)
        if "attention_mask" in data:
            data["attention_mask"] = data["attention_mask"].to(torch.int8)

    assert data["x_clean"].shape[1] == seq_len
    return data


def resolve_valid_dataset_dir(valid_dir, train_data_dir):
    if valid_dir is not None:
        valid_path = resolve_data_root(valid_dir)
        if not valid_path.is_dir():
            raise ValueError(f"valid_dir must be a dataset directory: {valid_path}")
        return valid_path

    train_path = resolve_data_root(train_data_dir)
    if train_path.is_dir():
        inferred = train_path.parent / "valid"
        if inferred.is_dir():
            return inferred
    raise ValueError(
        "Could not infer valid dataset dir. Please pass --valid_dir "
        "(e.g., qm9/parsed/base/valid)."
    )


def dict_to_dataset(data_dict):
    processed = {k: v.tolist() for k, v in data_dict.items()}
    dataset = Dataset.from_dict(processed)
    dataset.set_format(type="torch", columns=list(data_dict.keys()))
    return dataset


def reorder_tensor1(tensor1, tensor2, tensor3):
    mapping = {tuple(row.tolist()): i for i, row in enumerate(tensor2)}

    indices = []
    for row in tensor3:
        key = tuple(row.tolist())
        if key in mapping:
            indices.append(mapping[key])
        else:
            print(f"Warning: {key} not found in tensor2, skipping.")

    return tensor1[indices], indices



if __name__ == "__main__":
    import argparse
    import os 
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_type", type=str, default="qm9")
    parser.add_argument("--D", type=int, default=None, help="Override sequence length for selected dataset.")
    parser.add_argument("--V", type=int, default=None, help="Override vocabulary size for selected dataset.")
    parser.add_argument("--data_dir", type=str, default="qm9/parsed/base/train")
    parser.add_argument("--valid_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="qm9/parsed/pair")

    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_total", type=int, default=1)
    args = parser.parse_args()

    dataset_key = args.dataset_type.lower()
    if dataset_key not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS.keys()))
        raise ValueError(
            f"Unsupported --dataset_type={args.dataset_type}. "
            f"Supported: {supported}"
        )
    D = args.D if args.D is not None else DATASET_CONFIGS[dataset_key]["D"]
    V = args.V if args.V is not None else DATASET_CONFIGS[dataset_key]["V"]

    args.data_dir = args.data_dir.replace("qm9", f"{args.dataset_type}")
    args.save_dir = args.save_dir.replace("qm9", f"{args.dataset_type}")


    data_dir = resolve_data_root(args.data_dir)
    pair_dataset_dir = resolve_data_root(args.save_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    pair_dataset_dir.mkdir(parents=True, exist_ok=True)


    data = load_train_data(args.data_dir, D, dataset_key)

    import df_cuda
    X1 = data["x_clean"].to(torch.int64).to("cuda").contiguous()

    total_samples = data["x_clean"].shape[0]
    gpu_total = args.gpu_total
    gpu_idx = args.gpu_idx
    batch_size = args.batch_size

    base = total_samples // gpu_total
    remainder = total_samples % gpu_total

    if gpu_idx < remainder:
        start_idx = gpu_idx * (base + 1)
        end_idx = start_idx + (base + 1)
    else:
        start_idx = remainder * (base + 1) + (gpu_idx - remainder) * base
        end_idx = start_idx + base

    n_samples = end_idx - start_idx

    z0_list, z1_list = [], []
    pbar = tqdm(total=n_samples, desc=f"GPU {gpu_idx} Generating pairs")

    sample_idx = start_idx
    while sample_idx < end_idx:
        batch_end = min(sample_idx + batch_size, end_idx)
        z1_batch = data["x_clean"][sample_idx:batch_end].to(torch.int64).to("cuda")

        z0_batch = df_cuda.inversion(z1_batch, X1, tau=args.tau, num_steps=args.num_steps)

        z0_list.append(z0_batch)
        z1_list.append(z1_batch)

        sample_idx += batch_size
        pbar.update(batch_end - (sample_idx - batch_size))

    pbar.close()

    z0_list = torch.cat(z0_list, dim=0).to(torch.uint8).cpu()
    z1_list = torch.cat(z1_list, dim=0).to(torch.uint8).cpu()

    pair = {
        "x_prior": z0_list,
        "x_clean": z1_list,
    }
    torch.save(pair, data_dir / f"gpu_{gpu_idx}.pt")

    if len([f for f in os.listdir(data_dir) if f.startswith("gpu_") and f.endswith(".pt")]) == gpu_total:
        all_z0, all_z1 = [], []
        for i in range(gpu_total):
            try:
                pair = torch.load(data_dir / f"gpu_{i}.pt")
                all_z0.append(pair['x_prior'])
                all_z1.append(pair['x_clean'])
            except:
                print(f"GPU {gpu_idx} is not the last GPU, skipping merge.")
                sys.exit()
            
        merged_pair = {
            'x_prior': torch.cat(all_z0, dim=0),
            'x_clean': torch.cat(all_z1, dim=0)
        }
        assert merged_pair['x_prior'].shape[0] == data['x_clean'].shape[0]
        torch.save(merged_pair, data_dir / "train_pair.pt")
        for i in range(gpu_total):
            gpu_file = data_dir / f"gpu_{i}.pt"
            if gpu_file.exists():
                gpu_file.unlink()
                print(f"Removed {gpu_file}")

    else:
        print(f"GPU {gpu_idx} is not the last GPU, skipping merge.")
        sys.exit()

    print("Converting to datasets.Dataset...")
    train_data = data
    pair_data = merged_pair

    assert train_data['x_clean'].shape == pair_data['x_prior'].shape == pair_data['x_clean'].shape

    z0, indices = reorder_tensor1(pair_data['x_prior'], pair_data['x_clean'], train_data['x_clean'])

    assert z0.shape == pair_data['x_prior'].shape

    row_equal = (pair_data['x_clean'][indices] == train_data['x_clean']).all(dim=1)
    all_equal = row_equal.all()

    print("Row별 비교 결과:", row_equal)
    print("전체 동일?:", all_equal.item())

    if all_equal.item() == True:
        pair_data = {
            'x_prior': z0,
            'x_clean': train_data['x_clean'],
        }
        reserved_keys = {"x_prior", "x_clean", "input_ids"}
        for key, value in train_data.items():
            if key not in reserved_keys:
                pair_data[key] = value
        pair_dataset = dict_to_dataset(pair_data)
        pair_dataset.save_to_disk(str(pair_dataset_dir / "train"))
        print(f"Saved to {pair_dataset_dir}")

        valid_dataset = resolve_valid_dataset_dir(args.valid_dir, args.data_dir)
        shutil.copytree(str(valid_dataset), str(pair_dataset_dir / "valid"), dirs_exist_ok=True)
        print(f"Copied {valid_dataset} to {pair_dataset_dir / 'valid'}")

        
