#!/bin/bash
#SBATCH --job-name=KFNOISY_SLURM_JLEE
#SBATCH --output=./out/KFNOISY_SLURM_JLEE%A_%a.out
#SBATCH --error=./out/KFNOISY_SLURM_JLEE%A_%a.err
#SBATCH --array=1-28
cd /home/jlee/param_fit_PF_KFnoisy
module load MATLAB/R2018b
matlab -nodesktop -r slurm_run_fit_batch_KFnoisy
