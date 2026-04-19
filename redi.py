import json
import os

import datasets
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import utils

import wandb
from tqdm import tqdm
from datetime import datetime

from datasets import Dataset

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


def _dict_to_dataset(data_dict):
    processed = {k: v.tolist() for k, v in data_dict.items()}
    dataset = Dataset.from_dict(processed)
    dataset.set_format(type="torch", columns=list(data_dict.keys()))
    return dataset


def _load_dataset_split(dataset_root, split):
  """Load a split directory, supporting both {split} and {split}.pt layouts."""
  candidate_dirs = [
    os.path.join(dataset_root, split),
    os.path.join(dataset_root, f"{split}.pt"),
  ]
  for split_dir in candidate_dirs:
    if os.path.isdir(split_dir):
      return datasets.load_from_disk(split_dir), split_dir
  raise FileNotFoundError(
    f"Could not find dataset split '{split}' under {dataset_root}. "
    f"Tried: {candidate_dirs}")


def _generate_reflow_dataset(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Reflow Dataset Generation.')
  start_time = datetime.now()
  logger.info(f'start time: {start_time}')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  xT_list, x0_list, attention_mask_list = [], [], []
  pbar = tqdm(total=config.sampling.total_samples, desc='Generating samples')
  generated = 0

  while generated < config.sampling.total_samples:

      x0 = model.prior_sample(config.loader.eval_batch_size,
                              model.num_tokens)
      xT, x0 = model.sample_pairs(
        num_steps=config.sampling.steps, x0=x0)
      if tokenizer.pad_token_id is not None:
        attention_mask = (x0 != tokenizer.pad_token_id).to(torch.int64)
      else:
        attention_mask = torch.ones_like(xT)
      xT_list.append(xT.detach().clone())
      x0_list.append(x0.detach().clone())
      attention_mask_list.append(attention_mask.detach().clone())
      pbar.update(x0.shape[0])
      generated += x0.shape[0]

  xT = torch.cat(xT_list, dim=0)[:config.sampling.total_samples]
  x0 = torch.cat(x0_list, dim=0)[:config.sampling.total_samples]
  attention_mask = torch.cat(attention_mask_list, dim=0)[:config.sampling.total_samples]
  pbar.close()
  assert xT.shape == x0.shape == attention_mask.shape, \
    f'xT.shape: {xT.shape}, x0.shape: {x0.shape}, attention_mask.shape: {attention_mask.shape} should be the same'

  save_dir = config.eval.save_dir
  os.makedirs(save_dir, exist_ok=True)
  pair_data = {
    'x_prior': xT,
    'x_clean': x0,
    'attention_mask': attention_mask
  }
  dataset = _dict_to_dataset(pair_data)
  train_dir = os.path.join(save_dir, "train")
  dataset.save_to_disk(train_dir)
  logger.info(f'Reflow train dataset saved at: {train_dir}')

  valid_set, valid_src = _load_dataset_split(config.data.dataset_path, "valid")
  valid_dir = os.path.join(save_dir, "valid")
  valid_set.save_to_disk(valid_dir)
  logger.info(f'Validation dataset copied from {valid_src} to: {valid_dir}')

  config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
  with open(f"{save_dir}/config.json", "w") as f:
    json.dump(config_dict, f, indent=4)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  elif config.algo.name == 'distillation':
    diffusion_model = algo.Distillation
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  assert config.mode == 'gen_reflow_dataset', \
    f'redi.py only supports mode=gen_reflow_dataset, got {config.mode}'
  _generate_reflow_dataset(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer,
    logger=logger)


if __name__ == '__main__':
  main()
  wandb.finish()
  import sys
  sys.exit(0)
