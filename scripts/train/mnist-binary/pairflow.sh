#!/bin/bash
#SBATCH -J duo-lm1b                   # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

cmd="python -u -m main \
  loader.global_batch_size=256 \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  data=mnist_binary \
  data.add_mask_token=False \
  wandb.name=train-pairflow-mnist_binary \
  algo.backbone=unet \
  model=unet_small \
  algo=duo_base \
  model.length=784 \
  optim.lr=2e-4 \
  lr_scheduler=constant_warmup \
  lr_scheduler.num_warmup_steps=2_000 \
  trainer.val_check_interval=1.0 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=2_000 \
  trainer.max_steps=10_000 \
  trainer.val_check_interval=2_000 \
  +trainer.check_val_every_n_epoch=null \
  is_vision=True \
  pair=True  \
  data.dataset_path=data/preprocessed/mnist-binary/parsed/pair"

echo $cmd
eval $cmd