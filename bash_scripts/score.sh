#!/bin/bash

# command line args
MODEL=$1
DATASET=$2
SPLIT=$3

# default slurm args
TIME=0-2
MEM=15G
GRES="gpu:2080Ti:1"
CPUS=4
LOG=slurm-logs

ARGS="--time $TIME"
ARGS="$ARGS --mem $MEM"
ARGS="$ARGS --cpus-per-task $CPUS"
ARGS="$ARGS --gres $GRES"
ARGS="$ARGS --output $LOG/score-%j.out"
ARGS="$ARGS --export=ALL,MODEL=$MODEL,DATASET=$DATASET,SPLIT=$SPLIT"
if [[ -n $EXCLUDE_NODES ]]; then
    ARGS="$ARGS --exclude=$EXCLUDE_NODES"
fi

echo "ARGS: $ARGS"

sbatch --parsable $ARGS \
    bash_scripts/score.slurm
