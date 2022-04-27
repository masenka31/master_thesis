#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=18G

RATIOS=$1
DATASET=$2

module load Julia/1.6.4-linux-x86_64

julia --threads 3 self_classifier.jl $RATIOS $DATASET