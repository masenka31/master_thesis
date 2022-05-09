#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G

RATIOS=$1

module load Julia/1.6.4-linux-x86_64

julia --threads 2 genmodel.jl $RATIOS