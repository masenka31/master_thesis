#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G

RATIOS=$1
DATASET=$2

module load Julia/1.6.4-linux-x86_64

julia --threads 3 classifier.jl $RATIOS $DATASET