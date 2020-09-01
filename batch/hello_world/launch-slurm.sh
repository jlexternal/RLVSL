#!/bin/bash
#SBATCH --job-name=TEST_SLURM_JLEE
#SBATCH --output=./TEST_SLURM_JLEE%A_%a.out
#SBATCH --error=./TEST_SLURM_JLEE%A_%a.err
#SBATCH --array=1-10
cd /home/jlee/batch
module load MATLAB/R2018b
matlab -nodesktop -r slurm_hello_world.m
