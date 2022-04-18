#!/bin/bash
# This runs experiments

SCRIPT=$1
DATASET=$2
NUM_SAMPLES=$3
NUM_CONC=$4

LOG_DIR="${HOME}/logs/masters/MIProblems/$SCRIPT"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

# for ratios in 0.05 0.1 0.15 0.2
for ratios in 0.15 0.2
do
    # submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/${DATASET}_r=${ratios}_%A_%a.out" \
    ./${SCRIPT}.sh $ratios $DATASET
done