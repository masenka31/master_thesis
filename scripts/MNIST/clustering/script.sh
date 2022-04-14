#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

MODEL=$1
RATIOS=$2
FULL=$3

module load Julia/1.6.4-linux-x86_64

julia --threads 3 calculate_seeds.jl $MODEL $RATIOS $FULL