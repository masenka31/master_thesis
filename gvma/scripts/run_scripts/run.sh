#!/bin/bash
# This runs experiments

SCRIPT=$1
NUM_SAMPLES=$2
NUM_CONC=$3

LOG_DIR="${HOME}/logs/masters/$SCRIPT"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for ratios in 0.01 0.02 0.05 0.1 0.2
do
    # submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/r=${ratios}_%A_%a.out" \
    ./${SCRIPT}.sh $ratios
done