#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --partition=cpulong
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

RATIOS=$1
FULL=$2

module load Julia/1.6.4-linux-x86_64

julia M2.jl $RATIOS $FULL