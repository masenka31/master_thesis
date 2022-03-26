#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

SEED=$1
RATIOS=$2
ACTIVATION=$3
FULL=$4

module load Julia/1.6.4-linux-x86_64

julia classifier_triplet.jl $SEED $RATIOS $ACTIVATION $FULL