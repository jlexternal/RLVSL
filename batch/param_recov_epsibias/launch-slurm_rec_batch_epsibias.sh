#!/bin/bash
#SBATCH --job-name=EPSI_SLURM_JLEE
#SBATCH --output=./epsi/EPSI_SLURM_JLEE%A_%a.out
#SBATCH --error=./epsi/EPSI_SLURM_JLEE%A_%a.err
#SBATCH --array=1-100
cd /home/jlee/param_recov_epsibias
module load MATLAB/R2018b
matlab -nodesktop -r slurm_rec_batch_epsibias.m
