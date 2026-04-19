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
  loader.global_batch_size=1024 \
  loader.batch_size=1024 \
  loader.eval_batch_size=1024 \
  data=qm9 \
  wandb.name=train-udlm-qm9 \
  model=small \
  algo=duo_base \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1_000 \
  lr_scheduler.lr_min=3e-6 \
  trainer.val_check_interval=10_000 \
  +trainer.check_val_every_n_epoch=null \
  trainer.max_steps=50_000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
  training.compute_loss_on_pad_tokens=True \
  data.dataset_path=data/preprocessed/qm9/parsed/base"

echo $cmd
eval $cmd