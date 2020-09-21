taskID = getenv('SLURM_ARRAY_TASK_ID');
iter_num = str2num(taskID);
fn_rec_batch_statfit_epsibias(iter_num);