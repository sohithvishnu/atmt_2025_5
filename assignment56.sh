#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=avgbeam_translate_test.out

# Load correct GPU module (choose according to your HW)
module load a100

module load miniforge3
source activate atmt

PROJECT_DIR=/home/syacha/data/atmt_2025/cz-en
DEST_DIR=$PROJECT_DIR/data/prepared
CKPT_DIR=/home/syacha/data/atmt_2025/cz-en/checkpoints
MODEL_DIR=$PROJECT_DIR/tokenizers
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/joint-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/joint-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_avg.pt \
    --output cz-en/output_avg.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en 
