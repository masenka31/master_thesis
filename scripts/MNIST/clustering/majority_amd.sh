#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=amdfast
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

MODEL=$1
RATIOS=$2
FULL=$3

module load Julia/1.6.4-linux-x86_64

julia calculate_clusters.jl $MODEL $RATIOS $FULL