#!/bin/bash

# command line args
MODEL=$1
DATASET=$2
SPLIT=$3
if [[ $# -eq 4 ]]; then
    DEPD=$4
fi

# default slurm args
TIME=1-0
MEM=15G
GRES="gpu:1"
CPUS=4
LOG=slurm-logs

ARGS="--time $TIME"
ARGS="$ARGS --mem $MEM"
ARGS="$ARGS --cpus-per-task $CPUS"
ARGS="$ARGS --gres $GRES"
ARGS="$ARGS --exclude=$EXCLUDE"
ARGS="$ARGS --output $LOG/inf-loss-%j.out"
ARGS="$ARGS --export=ALL,MODEL=$MODEL,DATASET=$DATASET,SPLIT=$SPLIT"
# exclude EXCLUDE_NODES if exists
if [ -n "$EXCLUDE_NODES" ]; then
    ARGS="$ARGS --exclude=$EXCLUDE_NODES"
fi
# if dependency provided, add it to args
if [[ ! -z $DEPD ]]; then
    ARGS="$ARGS --dependency=afterany:$DEPD"
fi

echo "ARGS: $ARGS"

sbatch --parsable $ARGS \
    bash_scripts/inf.slurm
