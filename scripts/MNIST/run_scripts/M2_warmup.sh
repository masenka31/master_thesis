#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

RATIOS=$1
FULL=$2

module load Julia/1.6.4-linux-x86_64

julia --threads 3 M2_warmup.jl $RATIOS $FULL