#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=averaging_weight.out

# Load correct GPU module (choose according to your HW)
module load a100

module load miniforge3
source activate atmt

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Average Weights

python avg_checkpoints.py \
        --checkpoint-dir cz-en/checkpoints/ \
        --num-last 7 \
        --output cz-en/checkpoints/checkpoint_avg.pt

