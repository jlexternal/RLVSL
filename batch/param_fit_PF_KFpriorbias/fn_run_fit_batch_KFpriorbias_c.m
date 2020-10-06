function fn_run_fit_batch_KFpriorbias_c(ibatch)

% Parameter fitting code (batch) for the unbiased KF model for experiment RLVSL.
%
% Note: nbatch must be user-specified within the code before submitting to queue.
% 
% Requires: 1/ Subject data
%           2/ rpnormv.m
%           3/ fit_noisyKF_epsibias.m
%           4/ VBMC (Acerbi 2020)

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);

% ***** REQUIRED: Manual user input for proper functionality ***** %

    % NOTE: the name of the string must match the data file in 
    %       working directory
    load('subj_resp_rew_all.mat'); % load all relevant subject data
    nbatch = numel(subjlist); % number of total batches to be sent to SLURM; calculate this properly to distribute the output evenly
    
% **************************************************************** %

isubj = subjlist(ibatch);

ms = .55;       % sampling mean
vs = .07413^2;  % sampling variance

addpath('./vbmc');
addpath('./Toolboxes');

cfg = [];
cfg.vs = vs;
cfg.ms = ms;
cfg.nsmp = 1e3;

% ------------------ model options ------------------
cfg.sbias_cor   = true; % assume prior bias toward correct response
cfg.sbias_ini   = true; % assume initial prior bias
cfg.epsi        = 0; % no epsilon-greedy type bias
cfg.theta       = 0; % assume argmax
cfg.ksi         = 0; % assume no learning noise bias

% ------------------ options from fitting unbiased KF model on the novel/random condition ------------
cfg.cscheme = 'ths';    % qvs:Q-value sampling          or ths:Thompson sampling
cfg.lscheme = 'sym';    % ind:independent               or sym:symmetric
cfg.nscheme = 'upd';    % rpe:reward prediction errors  or upd:action value updates
% ----------------------------------------------------------------------------------------------------

cfg.verbose = true;

for icond = 1:2
    for iquar = 1:4
        blockrange = 4*(iquar-1)+1:4*(iquar-1)+4;
        cfg.resp = subj_resp_rew_all(isubj).resp(blockrange,:,icond);
        cfg.rt = subj_resp_rew_all(isubj).rew_seen(blockrange,:,icond);

        fprintf('Fitting subject %d over experimental condition %d on quarter %d\n',isubj,icond,iquar);
        
        out_fit{icond,iquar,isubj} = fit_noisyKF_epsibias(cfg); % fit the model and store
    end
end

if cfg.sbias_cor == true
    bias_type = 'biasCorr';
else
    bias_type = 'biasSubj';
end

bias_type

savename = sprintf('out_fit_KFpriorbias_%s_%s%s%s_%d_%02d',bias_type,cfg.cscheme(1),cfg.lscheme(1),cfg.nscheme(1),nbatch,isubj);
save(savename,'out_fit');

end