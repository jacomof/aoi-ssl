#!/bin/bash
# SLURM specific configs to run on a cluster. You may need to modify these based on your cluster's configuration.
#SBATCH -c 8         # number of cores            
#SBATCH -t 48:00:00    # time (HH:MM:SS)
#SBATCH --gres=gpu:1    # 1 indicates # of GPUs  
#SBATCH --mem=30gb
#SBATCH --output=logs/output/deeplab_output_%x-%J.out
#SBATCH --error=logs/error/deeplab_output_%x-%J.err


# You need to pass your conda location in order for slurm
# to access the right conda environment.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate aoi-ssl

# debugging flags (optional)
export PYTHONDONTWRITEBYTECODE=1

python -m segmentation.train --config ./configs/finetune/finetune_main.yml \
    --model_config ./configs/finetune/finetune_deeplab.yml \
    --model deeplab \
    --run_key "finetune_deeplab" \
    --path "./checkpoints/" \
    --data_path "./datasets/MNIST/finetune/" \