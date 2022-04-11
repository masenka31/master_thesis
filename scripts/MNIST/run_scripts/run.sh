#!/bin/bash
# This runs experiments

SCRIPT=$1

LOG_DIR="${HOME}/logs/masters/$SCRIPT"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

# submit n experiments to slurm
# with all arguments
# for n in {1..50}
# do
#     for ratios in 0.002 0.01 0.05
#     do
#         for full in 0 1
#         do
#             # submit to slurm
#             sbatch \
#             --output="${LOG_DIR}/${n}_%A_%a.out" \
#             ./${SCRIPT}.sh $ratios $full
#         done
#     done
# done

for n in {1..30}
do
    for ratios in 0.002 0.01
    do
        for full in 0 1
        do
            # submit to slurm
            sbatch \
            --output="${LOG_DIR}/n=${n}_r=${ratios}_f=${full}_%A_%a.out" \
            ./${SCRIPT}.sh $ratios $full
        done
    done
done