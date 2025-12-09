l#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=translate_test.out

# Load correct GPU module (choose according to your HW)
module load a100

module load miniforge3
source activate atmt

PROJECT_DIR=/home/syacha/data/atmt_2025/cz-en
DEST_DIR=$PROJECT_DIR/data/prepared
CKPT_DIR=/home/syacha/data/atmt_2025/cz-en/checkpoints
MODEL_DIR=$PROJECT_DIR/tokenizers
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# translate check for avg weights and noraml best weights 
  GNU nano 7.2                                                                    assignment54.sh *                                                                           
l#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=translate_test.out

# Load correct GPU module (choose according to your HW)
module load a100

module load miniforge3
source activate atmt

PROJECT_DIR=/home/syacha/data/atmt_2025/cz-en
DEST_DIR=$PROJECT_DIR/data/prepared
CKPT_DIR=/home/syacha/data/atmt_2025/cz-en/checkpoints
MODEL_DIR=$PROJECT_DIR/tokenizers
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

