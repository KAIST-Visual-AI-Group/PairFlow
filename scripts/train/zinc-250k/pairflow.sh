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
  data=zinc250k \
  wandb.name=train-pairflow-zinc250k \
  model=small \
  algo=duo_base \
  model.length=74 \
  optim.lr=3e-4 \
  trainer.val_check_interval=10_000 \
  +trainer.check_val_every_n_epoch=null \
  trainer.max_steps=200_000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20_000 \
  training.compute_loss_on_pad_tokens=True \
  data.dataset_path=data/preprocessed/zinc-250k/parsed/pair \
  pair=True"

echo $cmd
eval $cmd