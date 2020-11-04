Batch model parameter recovery for unbiased KF model using initial empirical prior with variable upper and lower asymptotes (RLVSL)

Execute steps in this order: 

-----------------------
0/ agglomerate_subj_data.m (from the main folder)
	-combines all subject data into a single data file to be used in the batch code

params_empirical_prior_noisyKF (data file with the Normal distribution parameters to act as empirical priors
-----------------------
1/ fn_run_fit_batch_KFpriorbias_empPrior.m
	-specify the name of the data file created above in this function
	-specify the fitting options (e.g. 'cfg.nscheme' for the noise scaling scheme)
	-specify the number of total batches to be run on the nodes (set to number of subjects)
2/ launch-slurm_run_fit_batch_KFpriorbias.sh
	-specify the upper limit of the SBATCH array to the same number as that in step 2
3/ Transfer contents to your folder in Frontex
4/ Send batch requests on SLURM using sbatch launch-slurm_run_fit_batch_KFpriorbias.sh
