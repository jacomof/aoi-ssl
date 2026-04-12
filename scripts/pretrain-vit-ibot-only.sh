#!/bin/bash
#SBATCH -p lanai   # partition (queue)
#SBATCH -c 6         # number of cores            
#SBATCH -t 100:00:00    # time (HH:MM:SS)
#SBATCH --gres=gpu:1    # 1 indicates # of GPUs  
#SBATCH --mem=30gb
#SBATCH --output=logs/output/%x-%J.out
#SBATCH --error=logs/error/%x-%J.err


# You need to pass your conda location in order for slurm
# to access the right conda environment.
source $HOME/miniforge3/etc/profile.d/conda.sh
conda init bash
conda activate aoi-ssl

# debugging flags (optional)
export PYTHONDONTWRITEBYTECODE=1


python -m pretrain.ibot.pretrain_ibot --config ./configs/pretrain/pretrain_main.yml \
    --model_config ./configs/pretrain/pretrain_vit_ibot_model_config.yml \
    --run_key "pretrain-vit-ibot" \
    --path "./checkpoints/" \
    --data_path "./datasets/MNIST/pretrain/" \
    --patience 100 \
    --monitor "val/loss" \
