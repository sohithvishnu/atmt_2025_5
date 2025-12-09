#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=train_joint-bpe.out

# Load correct GPU module (choose according to your HW)
module load a100

module load miniforge3
source activate atmt

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRAIN 
python train.py \
    --cuda \
    --data cz-en/data/prepared/ \
    --src-tokenizer cz-en/tokenizers/joint-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/joint-bpe-8000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 64 \
    --arch transformer \
    --max-epoch 7 \
    --log-file cz-en/logs/train.log \
    --save-dir cz-en/checkpoints/ \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 300 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3 \
    --epoch-checkpoints
