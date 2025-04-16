#!/bin/bash

#SBATCH --time 0-5
#SBATCH --mem 50G
#SBATCH --partition cpu
#SBATCH --output=slurm-logs/download-%j.out

export HF_HOME=huggingface

HF_TOKEN=$(cat tokens/hf_access_token.txt)
huggingface-cli login --token $HF_TOKEN

LOCAL_DIR=artifacts/huggingface

python src/download.py \
    --repo-id meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir $LOCAL_DIR \
    --repo-type model

python src/download.py \
    --repo-id meta-llama/Meta-Llama-3-70B-Instruct \
    --local-dir $LOCAL_DIR \
    --repo-type model

python src/download.py \
    --repo-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir $LOCAL_DIR \
    --repo-type model

python src/download.py \
    --repo-id meta-llama/Meta-Llama-3.1-70B-Instruct \
    --local-dir $LOCAL_DIR \
    --repo-type model

python src/download.py \
    --repo-id CohereForAI/c4ai-command-r-08-2024 \
    --local-dir $LOCAL_DIR \
    --repo-type model


python src/download.py \
    --repo-id ai21labs/AI21-Jamba-1.5-Mini \
    --local-dir $LOCAL_DIR \
    --repo-type model

python src/download.py \
    --repo-id Salesforce/SFR-Embedding-2_R \
    --local-dir $LOCAL_DIR \
    --repo-type model

python src/download.py \
    --repo-id dwzhu/e5rope-base \
    --local-dir $LOCAL_DIR \
    --repo-type model

python src/download.py \
    --repo-id Yale-LILY/a3cu \
    --local-dir $LOCAL_DIR \
    --repo-type model
