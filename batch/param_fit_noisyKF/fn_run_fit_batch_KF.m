function fn_run_fit_batch_KF(ibatch)

% Parameter fitting code (batch) for the biased KF model for experiment RLVSL.
%
% Note: nbatch must be user-specified within the code before submitting to queue.
% 
% Requires: 1/ Subject data
%           2/ rpnormv.m
%           3/ fit_noisyKF.m
%           4/ VBMC (Acerbi 2020)

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);

addpath('./vbmc');
addpath('./Toolboxes');

% ***** REQUIRED: Manual user input for proper functionality ***** %

    % NOTE: the name of the string must match the data file in 
    %       working directory
    load('subj_resp_rew_all.mat'); % load all relevant subject data
    nbatch = numel(subjlist); % number of total batches to be sent to SLURM; calculate this properly to distribute the output evenly
    
% **************************************************************** %

isubj = subjlist(ibatch);

ms = .55;       % sampling mean
vs = .07413^2;  % sampling variance

cfg = [];
cfg.vs = vs;
cfg.ms = ms;
% --- options from fitting unbiased KF model on the novel/random condition ---
cfg.cscheme = 'ths';    % qvs:Q-value sampling          or ths:Thompson sampling
cfg.lscheme = 'sym';    % ind:independent               or sym:symmetric
cfg.nscheme = 'upd';    % rpe:reward prediction errors  or upd:action value updates
% --- model options ---
cfg.epsi        = 0; % fit priorbias 
cfg.sbias_ini   = true;
cfg.ksi         = 0; % assume no learning noise bias
%cfg.theta       = 0; % fit softmax temperature
% --- hyperparameters (vbmc) ---
cfg.nsmp = 1e3;
cfg.nres = 1e3;
cfg.nval = 1e2;
cfg.nrun = 1;
cfg.verbose = true;
% --- experimental setting to fit to ---
conds       = 1:2;        % specify 1:rep / 2:alt / 3:rnd / or a range
fit_time    = 'qts';    % specify 'qts'/'all'
% --------------------------------------

if strcmpi(fit_time,'quarters')
    times = 1:4
else
    times = 5;
end
   
condstrs = {'rep','alt','rnd'};
condstr = '';
for icond = conds
    condstr = [condstr '_' condstrs{icond}];
    for itime = times
        if itime == 5
            blockrange = 1:16;
            timestr = sprintf('over all quarters');
        else
            blockrange = 4*(itime-1)+1:4*(itime-1)+4;
            timestr = sprintf('on quarter %d',itime);
        end
        cfg.resp    = subj_resp_rew_all(isubj).resp(blockrange,:,icond);
        cfg.rt      = subj_resp_rew_all(isubj).rew_seen(blockrange,:,icond);
        
        fprintf('Fitting subject %d over experimental condition %d %s\n',isubj,icond,timestr);
        
        out_fit{icond,itime,isubj} = fit_noisyKF(cfg); % fit the model and store
    end
end

savename = sprintf('out_fit_noisyKF%s_%d_%d',condstr,nbatch,isubj);
save(savename,'out_fit');
