"""
ZINC-250k preprocessing aligned with:
  iclr-2026-discrete-diffusion/fast-dfm/data/zinc250k/load_zinc250k.ipynb

- HuggingFace: ``yairschiff/zinc250k`` (splits ``train`` / ``validation``).
- Tokenizer: ``yairschiff/zinc250k-tokenizer`` (``trust_remote_code=True``).
- Tokenization matches the notebook: ``max_length=74``, ``padding=max_length``,
  ``canonical_smiles`` column.

Output matches ``qm9/download_dataset.py``: ``train.pt`` / ``valid.pt`` with
``x_clean`` + ``attention_mask``, and HF datasets under ``<output>/base/train``
and ``<output>/base/valid``. Sequence length is 74 (padded).
"""
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
import transformers


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


def save_zinc250k_tokenized_dataset(
    output_root=None,
    tokenizer_name="yairschiff/zinc250k-tokenizer",
    dataset_name="yairschiff/zinc250k",
    max_length=74,
    trust_remote_code=True,
    force_int8=True,
    num_proc=8,
):
    """
    Download ZINC-250k from HuggingFace (yairschiff/zinc250k), tokenize with
    yairschiff/zinc250k-tokenizer, and save like QM9: train.pt / valid.pt and
    HF datasets under parsed/base/{train,valid}.

    Splits follow the hub dataset: "train" and "validation" (no random split).
    """
    if output_root is None:
        output_root = SCRIPT_DIR / "parsed"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = SCRIPT_DIR / output_root

    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading ZINC-250k train / validation splits...")
    full = load_dataset(dataset_name)
    if "train" not in full or "validation" not in full:
        raise ValueError(
            f"Expected 'train' and 'validation' splits in {dataset_name}, got {list(full.keys())}"
        )
    train_ds = full["train"]
    valid_ds = full["validation"]

    if "canonical_smiles" not in train_ds.column_names:
        raise ValueError("canonical_smiles column not found in train split")
    if "canonical_smiles" not in valid_ds.column_names:
        raise ValueError("canonical_smiles column not found in validation split")

    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )

    def preprocess_and_tokenize(example):
        text = example["canonical_smiles"]

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        tokens = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        return tokens

    print("Tokenizing train...")
    train_tok = train_ds.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing train",
    )

    print("Tokenizing validation...")
    valid_tok = valid_ds.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing valid",
    )

    keep_cols = ["input_ids", "attention_mask"]
    train_tok = train_tok.remove_columns(
        [c for c in train_tok.column_names if c not in keep_cols]
    )
    valid_tok = valid_tok.remove_columns(
        [c for c in valid_tok.column_names if c not in keep_cols]
    )

    print("Converting to torch tensors...")
    train_input_ids = torch.tensor(train_tok["input_ids"], dtype=torch.long)
    train_attention_mask = torch.tensor(train_tok["attention_mask"], dtype=torch.long)

    valid_input_ids = torch.tensor(valid_tok["input_ids"], dtype=torch.long)
    valid_attention_mask = torch.tensor(valid_tok["attention_mask"], dtype=torch.long)

    max_token_id = max(train_input_ids.max().item(), valid_input_ids.max().item())
    min_token_id = min(train_input_ids.min().item(), valid_input_ids.min().item())
    print(f"token id range: [{min_token_id}, {max_token_id}]")

    if force_int8:
        if min_token_id < -128 or max_token_id > 127:
            raise ValueError(
                f"Token ids exceed int8 range: [{min_token_id}, {max_token_id}]. "
                "Set force_int8=False or use a larger integer dtype."
            )
        input_dtype = torch.int8
    else:
        input_dtype = (
            torch.int8
            if max_token_id <= 127 and min_token_id >= -128
            else torch.int16
        )

    train_data = {
        "x_clean": train_input_ids.to(input_dtype),
        "attention_mask": train_attention_mask.to(torch.int8),
    }
    valid_data = {
        "x_clean": valid_input_ids.to(input_dtype),
        "attention_mask": valid_attention_mask.to(torch.int8),
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
        "train_data_path": str(train_data_path),
        "valid_data_path": str(valid_data_path),
        "train_dataset_path": str(train_dataset_path),
        "valid_dataset_path": str(valid_dataset_path),
    }


if __name__ == "__main__":
    paths = save_zinc250k_tokenized_dataset(
        output_root=SCRIPT_DIR / "parsed",
        tokenizer_name="yairschiff/zinc250k-tokenizer",
        dataset_name="yairschiff/zinc250k",
        max_length=74,
        trust_remote_code=True,
        force_int8=True,
        num_proc=8,
    )
    print(paths)
