% run_pupil_analysis.m
% 
% Usage: Standard analysis where pupil areas are compared with the variables of the
%        incident trial

clear all;
addpath('./Toolboxes/'); % add path containing ANOVA function
%% Epoch raw data and detrend

% subjects to be included in analysis
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

% Condition to analyze
condtypes = ["rep","alt","rnd"];
nc = numel(condtypes);

epc_struct = {};

% Check pupil_get_epochs.m for optional config.
usecfg = true;
if usecfg
    cfg = struct;
    %cfg.polyorder = 20; % default is nt
    cfg.r_ep_lim = 4; % set the rightward limit of epoch window (in seconds)
    cfg.incl_nan = true;
    cfg.isdetrend = false;
    cfg.iszscored = true;
end

if usecfg
    epc_struct = pupil_get_epochs(subjlist,cfg);
else
    epc_struct = pupil_get_epochs(subjlist);
end


% choose the smallest epoch window for comparison
epc_range    = min([epc_struct.epoch_window],[],1);
epc_fb_onset = epc_struct.epoch_window(1) +1;

ifig = 0;

%%

%% Local functions
function rgb = graded_rgb(ic,iq,nq)
    xc = linspace(.8,.2,nq);

    rgb =  [1,xc(iq),xc(iq); ...
               xc(iq),1,xc(iq); ...
               xc(iq),xc(iq),1];

    rgb = rgb(ic,:);
end
