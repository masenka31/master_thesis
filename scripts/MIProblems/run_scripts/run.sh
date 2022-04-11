#!/bin/bash
# This runs experiments

SCRIPT=$1
DATASET=$2

LOG_DIR="${HOME}/logs/masters/MIProblems/$SCRIPT"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for n in {1..10}
do
    for ratios in 0.05 0.1 0.25
    # for ratios in 0.05
    do
        # submit to slurm
        sbatch \
        --output="${LOG_DIR}/${DATASET}n=${n}_r=${ratios}_%A_%a.out" \
        ./${SCRIPT}.sh $ratios $DATASET
    done
done