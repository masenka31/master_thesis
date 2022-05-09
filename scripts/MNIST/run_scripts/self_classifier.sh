#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G

RATIOS=$1
FULL=$2

module load Julia/1.6.4-linux-x86_64

julia --threads 3 self_classifier.jl $RATIOS $FULL