#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=benchmark_results2.out

# Load modules
module load a100
module load miniforge3
source activate atmt

# Setup environment variables
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# --- RUN BENCHMARK ---
# Note: We removed --bleu, --reference, and --output because 
# benchmark1.py only measures speed and prints to the console/log.

python benchmark2.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/joint-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/joint-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_avg.pt \
    --max-len 300 \
    --reference ~/shares/cz-en/data/raw/test.en
