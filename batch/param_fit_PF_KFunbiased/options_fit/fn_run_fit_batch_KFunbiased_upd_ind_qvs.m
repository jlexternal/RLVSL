function fn_run_fit_batch_KFunbiased(ibatch)

% Parameter fitting code (batch) for the unbiased KF model for experiment RLVSL.
%
% Note: nbatch must be user-specified within the code before submitting to queue.
% 
% Requires: 1/ Subject data
%           4/ fit_noisyKF_epsibias.m
%           5/ VBMC (Acerbi 2020)

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);

% ***** REQUIRED: Manual user input for proper functionality ***** %

    % NOTE: the name of the string must match the data file in 
    %       working directory
    load('subj_resp_rew_all.mat'); % load all relevant subject data
    icond = 3; 
    nbatch = numel(subjlist); % number of total batches to be sent to SLURM; calculate this properly to distribute the output evenly
    
% **************************************************************** %

isubj = subjlist(ibatch);

ms = .55;       % sampling mean
vs = .07413^2;  % sampling variance

addpath('./vbmc');
addpath('./Toolboxes');

cfg = [];
cfg.resp = subj_resp_rew_all(isubj).resp(:,:,icond);
cfg.rt = subj_resp_rew_all(isubj).rew_seen(:,:,icond);
cfg.vs = vs;
cfg.ms = ms;
cfg.nsmp = 1e3;
cfg.epsi    = 0; % no epsilon-bias
cfg.theta   = 0; % argmax
cfg.ksi     = 0; % no learning noise bias
cfg.nscheme = 'upd';    % rpe:reward prediction errors  or upd:action value updates
cfg.lscheme = 'ind';    % ind:independent               or sym:symmetric
cfg.cscheme = 'qvs';    % qvs:Q-value sampling          or ths:Thompson sampling
cfg.sbias_cor = 'false';
cfg.sbias_ini = 'false';
cfg.verbose = true;

out_fit{icond,isubj} = fit_noisyKF_epsibias(cfg); % fit the model


savename = sprintf('out_fit_KFunbiased_%s_%s_%s_%d_%02d',cfg.cscheme,cfg.lscheme,cfg.nscheme,nbatch,isubj);
save(savename,'out_fit');

end