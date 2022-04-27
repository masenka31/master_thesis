#!/bin/bash
# This runs experiments

LOG_DIR="${HOME}/logs/masters/clustering"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi
#classifier_triplet self_classifier M2 M2_warmup
for modelname in M2 M2_warmup
do
    # for ratios in 0.002 0.01 0.05 0.1
    for ratios in 0.002 0.01 0.05 0.1
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