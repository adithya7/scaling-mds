#!/bin/bash

# command line args
MODEL=$1
DATASET=$2
SPLIT=$3
if [[ $# -eq 4 ]]; then
    DEPD=$4
fi

# default slurm args
TIME=2-0
MEM=140G
# if model starts with Llama3_8B or Llama31_8B, use 1 gpu
# else use 4 gpus
if [[ $MODEL == Llama31_8B* || $MODEL == Llama3_8B* ]]; then
    GRES="gpu:1"
    MEM=90G
else
    GRES="gpu:4"
fi
CPUS=4
LOG=slurm-logs

ARGS="--time $TIME"
ARGS="$ARGS --mem $MEM"
ARGS="$ARGS --cpus-per-task $CPUS"
ARGS="$ARGS --gres $GRES"
ARGS="$ARGS --exclude=$EXCLUDE"
ARGS="$ARGS --output $LOG/pred-%j.out"
ARGS="$ARGS --export=ALL,MODEL=$MODEL,DATASET=$DATASET,SPLIT=$SPLIT"
# if dependency provided, add it to args
if [[ ! -z $DEPD ]]; then
    ARGS="$ARGS --dependency=afterany:$DEPD"
fi

echo "ARGS: $ARGS"

sbatch --parsable $ARGS \
    bash_scripts/pred.slurm
