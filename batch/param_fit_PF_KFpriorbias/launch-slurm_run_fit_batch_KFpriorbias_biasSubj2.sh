#!/bin/bash
#SBATCH --job-name=KFPRIORBIAS_S_SLURM_JLEE
#SBATCH --output=./out_s/KFPRIORBIAS_S_SLURM_JLEE%A_%a.out
#SBATCH --error=./out_s/KFPRIORBIAS_S_SLURM_JLEE%A_%a.err
#SBATCH --array=1-28
cd /home/jlee/param_fit_PF_KFpriorbias
module load MATLAB/R2018b
matlab -nodesktop -r slurm_run_fit_batch_KFpriorbias_biasSubj2
