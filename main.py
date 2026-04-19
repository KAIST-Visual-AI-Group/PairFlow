import json
import os

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

from PIL import Image

import rdkit
rdkit.rdBase.DisableLog('rdApp.error')
from rdkit import Chem as rdChem
from rdkit.Chem import QED
import datasets
import typing

import wandb
from tqdm import tqdm
import numpy as np


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


def _eval_fid(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting FID Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  all_samples = []
  pbar = tqdm(total=config.sampling.total_samples, desc='Generating samples')
  while len(all_samples) < config.sampling.total_samples:
    samples = model.restore_model_and_sample(
      num_steps=config.sampling.steps)
    images = model.tokenizer.batch_decode(samples)
    all_samples.extend(list(images))
    pbar.update(len(images))

  samples_path = config.eval.generated_samples_path
  os.makedirs(samples_path, exist_ok=True)
  for i, image in enumerate(all_samples):
    if i >= config.sampling.total_samples:
      break
    image = image.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(f'{samples_path}/{i:0>5}.png')
  print('Samples saved at:', samples_path, f'{i} images')

  if config.data.name == 'cifar10':
    from pytorch_image_generation_metrics import get_inception_score_and_fid_from_directory
    (is_score, IS_std), fid_score = get_inception_score_and_fid_from_directory(
        samples_path,
        './fid_features/cifar-10.npz')

    print(f'\n=============== CIFAR10 ================')
    print(f'checkpoint path: {config.eval.checkpoint_path}')
    print(f'Samples path: {samples_path}')
    print(f'num_steps: {config.sampling.steps} | total_samples: {config.sampling.total_samples}')
    print(f'[FID] score: {fid_score}')
    print(f'[IS] score: {is_score}')
    print(f'========================================')

  elif config.data.name == 'mnist_binary':
    from pytorch_image_generation_metrics import get_inception_score_and_fid_from_directory
    (is_score, IS_std), fid_score = get_inception_score_and_fid_from_directory(
        samples_path,
        './fid_features/mnist-binary.npz')

    print(f'\n=============== MNIST-BINARY ================')
    print(f'checkpoint path: {config.eval.checkpoint_path}')
    print(f'Samples path: {samples_path}')
    print(f'num_steps: {config.sampling.steps} | total_samples: {config.sampling.total_samples}')
    print(f'[FID] score: {fid_score}')
    print(f'=============================================')
  else:
    raise ValueError(f'Invalid data: {config.data.name}')

  metadata = {
    'checkpoint_path': config.eval.checkpoint_path,
    'samples_path': samples_path,
    'num_steps': config.sampling.steps,
    'total_samples': config.sampling.total_samples,
    'fid_score': fid_score,
    'is_score': is_score,
  }
  with open(f'{samples_path}/_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)


def _eval_qm9(diffusion_model, config, logger, tokenizer):
  logger.info('Starting QM9 Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)

  qm9_dataset = datasets.load_dataset(
    'yairschiff/qm9', split='train')

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  def get_mol_property_fn(
      prop: str
  ) -> typing.Callable[[rdChem.Mol], typing.Union[int, float]]:
    if prop == 'qed':
      return QED.qed
    if prop == 'ring_count':
      return lambda x_mol: len(rdChem.GetSymmSSSR(x_mol))
    raise NotImplementedError(
      f"Property function for {prop} not implemented")


  valids_tot, unique_tot, novel_tot, invalid_tot = [], [], [], []
  valids_pct_tot, unique_pct_tot, novel_pct_tot, invalid_pct_tot = [], [], [], []

  if getattr(config.sampling, 'num_trials', None) is None:
    num_trials = 1
  else:
    num_trials = config.sampling.num_trials

  for _ in tqdm(range(num_trials), desc='Generating samples'):
    all_samples = []
    while len(all_samples) < config.sampling.total_samples:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      all_samples.extend(list(text_samples))


    raw_samples = all_samples.copy()

    mol_property_fn = get_mol_property_fn(config.eval.label_col)
    all_samples = [(seq + '<eos>').split('<eos>')[0] for seq in all_samples]
    all_samples = [seq.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '') for seq in all_samples]
    invalids, valids, mol_property = [], [], []
    for i, seq in enumerate(all_samples):
      if i >= config.sampling.total_samples:
        break
      try:
        seq = seq.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')
        mol = rdChem.MolFromSmiles(seq)
        if mol is None or len(seq) == 0:
          invalids.append(seq)
        else:
          valids.append(seq)
          mol_property.append(mol_property_fn(mol))
      except rdkit.Chem.rdchem.KekulizeException as e:
        print(e)
        invalids.append(seq)

    invalid = len(invalids)
    invalid_pct = invalid / len(all_samples)
    valid = len(valids)
    valid_pct = valid / len(all_samples)
    unique = len(set(valids))
    novel = len(set(valids) - set(qm9_dataset['canonical_smiles']))
    try:
      unique_pct = unique / valid
      novel_pct = novel / valid
    except ZeroDivisionError:
      unique_pct, novel_pct = 0., 0.

    valids_tot.append(valid)
    unique_tot.append(unique)
    novel_tot.append(novel)
    valids_pct_tot.append(valid_pct)
    unique_pct_tot.append(unique_pct)
    novel_pct_tot.append(novel_pct)
    invalid_tot.append(invalid)
    invalid_pct_tot.append(invalid_pct)

  samples_path = config.eval.generated_samples_path
  with fsspec.open(f'{samples_path}/all_samples.json', 'w') as f:
    json.dump({'total_samples': config.sampling.total_samples,
               'generated_seqs': all_samples,
               'raw_samples': raw_samples}, f, indent=4)
  print('Samples saved at:', samples_path)

  print(f'\n=============== QM9 ================')
  print(f'checkpoint path: {config.eval.checkpoint_path}')
  print(f'Samples path: {samples_path}')
  print(f'num_steps: {config.sampling.steps} | total_samples: {config.sampling.total_samples}')
  print(f'Valid: {np.mean(valids_tot)} / {len(all_samples)} ({100 * np.mean(valids_pct_tot):0.2f}%) ')
  print(f'Unique (of valid): {np.mean(unique_tot)} / {np.mean(valids_tot)} ({100 * np.mean(unique_pct_tot):0.2f}%) ')
  print(f'Novel (of valid): {np.mean(novel_tot)} / {np.mean(valids_tot)} ({100 * np.mean(novel_pct_tot):0.2f}%)')
  print(f'invalid: {np.mean(invalid_tot)} / {len(all_samples)} ({100 * np.mean(invalid_pct_tot):0.2f}%) ')
  print(f'========================================')

  metadata = {
    'checkpoint_path': config.eval.checkpoint_path,
    'samples_path': samples_path,
    'num_steps': config.sampling.steps,
    'total_samples': config.sampling.total_samples,
    'valid_tot': np.mean(valids_tot).item(),
    'valid_std': np.std(valids_tot).item(),
    'unique_tot': np.mean(unique_tot).item(),
    'unique_std': np.std(unique_tot).item(),
    'novel_tot': np.mean(novel_tot).item(),
    'novel_std': np.std(novel_tot).item(),
    'invalid_tot': np.mean(invalid_tot).item(),
    'invalid_std': np.std(invalid_tot).item(),
  }
  with open(f'{samples_path}/_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config)
  else:
    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
  trainer.logger.experiment.finish()


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
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
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'fid_eval':
    _eval_fid(**kwargs)
  elif config.mode == 'qm9_eval':
    _eval_qm9(**kwargs)
  else:
    _train(**kwargs)

  wandb.finish()


if __name__ == '__main__':
  main()
