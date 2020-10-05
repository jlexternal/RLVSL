#!/bin/bash
#SBATCH --job-name=KFUNBIAS_USQ_SLURM_JLEE
#SBATCH --output=./out/KFUNBIAS_USQ_SLURM_JLEE%A_%a.out
#SBATCH --error=./out/KFUNBIAS_USQ_SLURM_JLEE%A_%a.err
#SBATCH --array=1-28
cd /home/jlee/param_fit_PF_KFunbiased
module load MATLAB/R2018b
matlab -nodesktop -r slurm_run_fit_batch_KFunbiased_upd_sym_qvs
