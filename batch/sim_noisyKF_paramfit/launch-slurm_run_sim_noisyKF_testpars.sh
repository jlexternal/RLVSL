#!/bin/bash
#SBATCH --job-name=NOISYKF_SIM_SLURM_JLEE
#SBATCH --output=./out/NOISYKF_SIM_SLURM_JLEE%A_%a.out
#SBATCH --error=./out/NOISYKF_SIM_SLURM_JLEE%A_%a.err
#SBATCH --array=1-28
cd /home/jlee/sim_noisyKF_paramfit
module load MATLAB/R2018b
matlab -nodesktop -r slurm_run_sim_noisyKF_testpars
