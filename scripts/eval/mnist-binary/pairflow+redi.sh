#!/bin/bash
#SBATCH -J owt_duo_anneal                    # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

SAMPLING_STEPS=""
TOTAL_SAMPLES=""
SAMPLING_TYPE="greedy"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --SAMPLING_STEPS)
      SAMPLING_STEPS="$2"; shift 2 ;;
    --TOTAL_SAMPLES)
      TOTAL_SAMPLES="$2"; shift 2 ;;
    --SAMPLING_TYPE)
      SAMPLING_TYPE="$2"; shift 2 ;;
    --*)
      shift 2 ;;
    *)
      if [[ -z "$SAMPLING_STEPS" ]]; then SAMPLING_STEPS="$1"; shift; continue; fi
      if [[ -z "$TOTAL_SAMPLES" ]]; then TOTAL_SAMPLES="$1"; shift; continue; fi
      if [[ -z "$SAMPLING_TYPE" ]]; then SAMPLING_TYPE="$1"; shift; continue; fi
      shift ;;
  esac
done

if [[ -z "$SAMPLING_STEPS" || -z "$TOTAL_SAMPLES" ]]; then
  echo "Missing required args: SAMPLING_STEPS and TOTAL_SAMPLES" >&2
  exit 1
fi

checkpoint_path=checkpoints/mnist-binary/pairflow+redi.ckpt
samples_path=./evaluation/mnist-binary/pairflow+redi-${SAMPLING_TYPE}
samples_path="$samples_path/num_steps-${SAMPLING_STEPS}_total-${TOTAL_SAMPLES}"

export HYDRA_FULL_ERROR=1

cmd="python -u -m main \
  mode=fid_eval \
  loader.eval_batch_size=256 \
  data=mnist_binary \
  model=unet_small \
  algo.backbone=unet \
  algo=duo_base \
  model.length=784 \
  sampling.steps=$SAMPLING_STEPS \
  sampling.noise_removal=$SAMPLING_TYPE \
  +sampling.total_samples=$TOTAL_SAMPLES \
  eval.checkpoint_path=$checkpoint_path \
  eval.generated_samples_path=$samples_path \
  is_vision=True \
  pair=True \
  sampling.predictor=ancestral \
  +wandb.offline=true"

echo $cmd
eval $cmd