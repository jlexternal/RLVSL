taskID = getenv('SLURM_ARRAY_TASK_ID');
iter_num = str2num(taskID);
fn_run_fit_batch_KFpriorbias_s2(iter_num);
