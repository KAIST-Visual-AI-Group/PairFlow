#!/bin/bash
#SBATCH -J posterior                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export HYDRA_FULL_ERROR=1
export WANDB__SERVICE_WAIT=10
finetune_path=checkpoints/mnist-binary/udlm.ckpt

cmd="python -u -m main \
  mode=train \
  loader.global_batch_size=256 \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  data=mnist_binary \
  data.add_mask_token=False \
  wandb.name=distill-udlm-redi \
  algo.backbone=unet \
  model=unet_small \
  model.length=784 \
  algo=duo_base \
  lr_scheduler=cosine_decay_warmup \
  training.finetune_path=$finetune_path \
  sampling.steps=128 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1_000 \
  trainer.val_check_interval=1_000 \
  +trainer.check_val_every_n_epoch=null \
  trainer.max_steps=5_000 \
  training.ema=0.999 \
  +algo.update_teacher_every=1_000 \
  +algo.teacher_ema=False \
  +algo.linear_growth_dt=false \
  optim.lr=6e-5 \
  trainer.limit_val_batches=8 \
  data.dataset_path=data/redi/mnist-binary/udlm \
  pair=True \
  is_vision=True"

echo $cmd
eval $cmd
