% run_mb_pupil_phasic_analysis.m
% 
% Usage: Phasic pupil analysis (detrended pupil analysis) on model-based variables
%        (from best-fitting parameters of a given model).
%
% Jun Seok Lee <jlexternal@gmail.com>
clear all;
close all;

%% Load and detrend pupil data

% import subject data
nsubjtot = 31;
excluded = [1 23 28];
subjlist = setdiff(1:nsubjtot, excluded); % remove excluded subjects
% load experiment structure
nsubj = numel(subjlist);
% Conditions to analyze
condtypes = ["rep","alt","rnd"];
nc = numel(condtypes);
epc_struct = {};
% Check pupil_get_epochs.m for optional config.
usecfg = true;
if usecfg
    cfg = struct;
    %cfg.polyorder = 20; % default is nt
    cfg.r_ep_lim = 4; % set the rightward limit of epoch window (in seconds)
    cfg.incl_nan = false;
    cfg.lim_epoch   = 'END';
    cfg.ievent      = 1; % 1/STIM 2/RESP 3/FBCK 4/END
    cfg.ievent_end  = 4;
end
if usecfg
    epc_struct = pupil_get_epochs(subjlist,cfg);
else
    epc_struct = pupil_get_epochs(subjlist);
end
% choose the smallest epoch window for comparison
epc_range    = min([epc_struct.epoch_window],[],1); % [nSamples before onset, nSamples after onset]
epc_fb_onset = epc_struct.epoch_window(1) +1;

%% Import model-based data



%% Organize: Regression of pupil dilation to chosen model-based variables

% model-based variable of choice
var_str = '';
var_t = [];
switch var_str
    case 'mt'
        
    case 'rpe'
        
        % refer to Rouhani et al. (2020) Reward prediction errors create event boundaries in memory
        
        
        
        
end

% model-free variable
% reward

% run logistic regression with all variables (seen reward, tracked/expected value, rpe)
