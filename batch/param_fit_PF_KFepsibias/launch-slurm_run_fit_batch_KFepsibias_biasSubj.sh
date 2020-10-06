#!/bin/bash
#SBATCH --job-name=KFEPSIBIAS_S_SLURM_JLEE
#SBATCH --output=./out_s/KFEPSIBIAS_S_SLURM_JLEE%A_%a.out
#SBATCH --error=./out_s/KFEPSIBIAS_S_SLURM_JLEE%A_%a.err
#SBATCH --array=1-28
cd /home/jlee/param_fit_PF_KFepsibias
module load MATLAB/R2018b
matlab -nodesktop -r slurm_run_fit_batch_KFepsibias_biasSubj
