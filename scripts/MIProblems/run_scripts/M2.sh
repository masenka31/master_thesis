#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=cpufast
#SBATCH --cpus-per-task=1
#SBATCH --mem=18G

RATIOS=$1

module load Julia/1.6.4-linux-x86_64

julia --threads 3 M2.jl $RATIOS