#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G

RATIOS=$1

module load Julia/1.6.4-linux-x86_64

julia --threads 5 self_classifier.jl $RATIOS