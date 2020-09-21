Batch model parameter recovery for epsilon-bias model (RLVSL)

Execute steps in this order: 

1/ sim_batch_epsibias.m 
	-run to create a data file called data_sim_epsibias_DDMMYYYY.mat
2/ fn_rec_batch_epsibias.m
	-specify the name of the data file created above in this function
	-specify the number of total batches to be run on the nodes
3/ launch-slurm_rec_batch_epsibias.sh
	-specify the upper limit of the SBATCH array to the same number as that in step 2

4/ Transfer contents of param_recov_epsibias to your folder in Frontex
5/ Send batch requests on SLURM using sbatch launch-slurm_rec_batch_epsibias.sh
