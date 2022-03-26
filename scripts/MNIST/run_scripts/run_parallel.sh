#!/bin/bash
# This runs experiments

SCRIPT=$1

LOG_DIR="${HOME}/logs/masters/$SCRIPT"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

# submit 10 experiments to slurm
# with all arguments
for n in {1..3}
do
    for seed in {1..5}
    do
        for ratios in 0.002 0.01 0.05
        do
            for activation in swish relu
            do
                for full in 0 1
                do
                    # submit to slurm
                    sbatch \
                    --output="${LOG_DIR}/${n}_%A_%a.out" \
                    ./${SCRIPT}.sh $seed $ratios $activation $full
                done
            done
        done
    done
done