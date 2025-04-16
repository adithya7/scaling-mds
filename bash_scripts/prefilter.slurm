#!/bin/bash

#SBATCH --time 2-0
#SBATCH --mem 70G
#SBATCH --gres gpu:L40:1
#SBATCH --output slurm-logs/prefilter-%j.out

RETRIEVER=$1

HF_TOKEN=$(cat tokens/hf_access_token.txt)
huggingface-cli login --token $HF_TOKEN

ARTIFACTS=artifacts
# this is needed for HF dataset loading scripts
export HF_DATA_PATH=artifacts

python src/prefilter.py \
    --retriever-config-name $RETRIEVER \
    --dataset-config-name BackgroundQFSumm \
    --split test \
    --max-segment-tokens 1024 \
    --max-input-tokens 65536 \
    --artifacts-dir $ARTIFACTS
