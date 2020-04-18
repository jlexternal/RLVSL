function gen_expe_subj_rlvsl(subj)
%
%  Usage: [expe] = GEN_EXPE_SUBJ_RLVSL(subj)
%
%  where subj is the subject number
%
%  Requires: gen_expe_rlvsl(subj)

% initialize random number generator
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle')); 

% generate experiment structure
expe = gen_expe_rlvsl(subj);

% save experiment structure
mkdir(sprintf('./Data/S%02d',subj));
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',subj,subj);
save(filename,'expe');


end