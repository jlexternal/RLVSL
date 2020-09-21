#!/bin/bash
#SBATCH --job-name=EPSISTAT_SLURM_JLEE
#SBATCH --output=./out/EPSISTAT_SLURM_JLEE%A_%a.out
#SBATCH --error=./out/EPSISTAT_SLURM_JLEE%A_%a.err
#SBATCH --array=1-49
cd /home/jlee/param_recov_statfit_epsibias
module load MATLAB/R2018b
matlab -nodesktop -r slurm_rec_batch_statfit_epsibias
