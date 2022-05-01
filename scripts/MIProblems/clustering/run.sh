#!/bin/bash
# This runs experiments

SCRIPT=$1

LOG_DIR="${HOME}/logs/masters/clustering"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for modelname in classifier classifier_triplet M2 self_classifier self_arcface
do
    for ratios in 0.05 0.1 0.15 0.2
    do
        # submit to slurm
        sbatch \
        --output="${LOG_DIR}/${modelname}_%A_%a.out" \
        ./${SCRIPT}.sh $modelname $ratios
    done
done