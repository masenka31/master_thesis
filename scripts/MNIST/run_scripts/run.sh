#!/bin/bash
# This runs experiments

SCRIPT=$1
NUM_SAMPLES=$2
NUM_CONC=$4

LOG_DIR="${HOME}/logs/masters/$SCRIPT"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for ratios in 0.002 0.01
do
    for full in 0 1
    do
        # submit to slurm
        sbatch \
        --array=1-${NUM_SAMPLES}%${NUM_CONC} \
        --output="${LOG_DIR}/n=${n}_r=${ratios}_f=${full}_%A_%a.out" \
        ./${SCRIPT}.sh $ratios $full
    done
done