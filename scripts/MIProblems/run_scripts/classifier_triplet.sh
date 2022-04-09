#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

RATIOS=$1

module load Julia/1.6.4-linux-x86_64

julia --threads 3 classifier_triplet.jl $RATIOS