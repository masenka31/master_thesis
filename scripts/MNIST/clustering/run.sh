#!/bin/bash
# This runs experiments

LOG_DIR="${HOME}/logs/masters/clustering"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for modelname in classifier classifier_triplet
do
    for ratios in 0.002 0.01 0.05
    do
        for full in 0 1
        do
            # submit to slurm
            sbatch \
            --output="${LOG_DIR}/${modelname}_%A_%a.out" \
            ./script.sh $modelname $ratios $full
        done
    done
done