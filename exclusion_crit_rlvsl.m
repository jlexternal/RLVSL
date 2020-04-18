% exclusion_crit_rlvsl.m

% Exclusion criteria analysis based on performance to the task compared to random
% choices

clear all;
close all;

ifig = 1;
nsubjtot    = 31;
excluded    = [1];
subjlist    = setdiff(1:nsubjtot, excluded);
% load experiment structure
nsubj = numel(subjlist);
% Data manip step
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',2,2);
load(filename,'expe');
nt = expe(1).cfg.ntrls;
nb = expe(1).cfg.nbout;
ntrain = expe(1).cfg.ntrain;
nb_c = nb/3;

sesh_acc = nan(nsubj, 8);

for isubj = subjlist
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj);
    fprintf('Loading subject %d...\n',isubj);
    load(filename,'expe');
    ib_c = ones(3,1);
    resps = zeros(nb,nt);
    isesh = 1;
    for ib = 1+ntrain:nb+ntrain
        resps(ib-ntrain,:) = -(expe(ib).resp-2);
        
        if mod(ib-ntrain,6) == 0 % 1 complete session
            sesh_acc(isubj,isesh) = mean(mean(resps(ib-ntrain-5:ib-ntrain,:)));
            
            rand_acc = mean(mean(round(rand(6,nt))));
            if sesh_acc(isubj,isesh) <= rand_acc
                fprintf('Subj number : %d underperformed less than random on session %d\n',isubj,isesh);
            end
            isesh = isesh + 1;
        end
    end
end