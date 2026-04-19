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

export HYDRA_FULL_ERROR=1
finetune_path=checkpoints/cifar-10/udlm.ckpt

cmd="python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  data=cifar10 \
  wandb.name=distill-udlm-dcd \
  algo.backbone=unet \
  model=unet \
  algo=distillation \
  lr_scheduler=cosine_decay_warmup \
  training.finetune_path=$finetune_path \
  model.length=3072 \
  trainer.val_check_interval=5_000 \
  +trainer.check_val_every_n_epoch=null \
  trainer.max_steps=50_000 \
  training.ema=0.999 \
  algo.update_teacher_every=10_000 \
  optim.lr=6e-5 \
  trainer.limit_val_batches=8 \
  algo.teacher_ema=False \
  algo.linear_growth_dt=false \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
  training.compute_loss_on_pad_tokens=True \
  data.dataset_path=data/preprocessed/cifar-10/parsed/base \
  is_vision=True"

echo $cmd
eval $cmd