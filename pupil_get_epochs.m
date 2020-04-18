function [out] = pupil_get_epochs(subjlist,condtype,cfg)
% pupil_get_epochs
%
% Usage: Organizes and detrends preprocessed pupil data into epochs for all included 
%           subjects to be analyzed for experiment RLVSL.
%
% Input:        subjlist - array of subjects to be included in analysis
%               condtype - experimental condition
%               cfg      - structure containing fields:
%    (optional)             lim_epoch : right-most limit of an epoch (STIM or END)
%    (optional)             min_prefb : minimum sampling points a trial must have before
%                                        feedback onset      
%    (optional)             incl_nan  : option to include epochs where NaN were found
%    (optional)             polyorder : order of polynomial for detrending algorithm
%
% Output: out - structure containing fields:
%                   fbs      : feedback values
%                   epochs   : detrended pupil data within epochs
%                   idx_subj : subject id for fb and pupil trends 
%                               (relative id, not absolute!)
%                   
% Jun Seok Lee <jlexternal@gmail.com>

if nargin < 3
    cfg = [];
end
if nargin < 2
    error('Provide condition to analyze over!');
end
if nargin < 1
    error('Missing subject list and experimental condition!');
end
if ~isfield(cfg,'incl_nan')
    cfg.incl_nan = false;
end

% Add robust detrending toolbox path
addpath('./Toolboxes/NoiseTools/');

% Find the epoch window that fits all subjects
[epoch_window,excluded_trials,fbs,qts] = pupil_epoch_window_rlvsl(subjlist,condtype,cfg);

fbs = fbs-50; % make the negative feedback actually negative
nt = size(fbs,1);
nb = size(fbs,2);
n_excld = sum(sum(sum(excluded_trials)));
epochs = nan(numel(fbs)-n_excld,sum(epoch_window)+1);
idx_subj_epoch = zeros(numel(fbs)-n_excld,1);

if ~isfield(cfg,'polyorder')
    cfg.polyorder = nt;
end

% Organize the data
ctr_epoch = 0;
ctr_subj = 0;
for isubj = subjlist
    ctr_subj = ctr_subj+1;
    load(sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)); % loads structure 'expe' to workspace
    
    ctr_blck = 0;
    for ib = 1:length(expe)
        if strcmpi(expe(ib).type,condtype)
            ctr_blck = ctr_blck+1;
            
            % Load the pupil data
            pupilfile = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b%02d*preproc.mat',isubj,isubj,ib));
            load(sprintf('./Data/S%02d/%s',isubj,pupilfile.name)); % loads structure 'data_eye' to workspace
            tsmp = data_eye.tsmp;
            psmp = data_eye.psmp;
            tmsg = data_eye.tmsg;
            
            % Detrend the data
                % Normalize the time sample points
                t_init = tsmp(1);
                tsmp_normd = (tsmp-t_init)/1000;

                % Identify NaN in data
                ind_nan = isnan(tsmp_normd);
                ind_nan = ind_nan | isnan(psmp);
                
                % Detrend data with Robust Detrending (of Alain de Cheveigné)
                pupil_detrend_rbst = nan(size(psmp));
                [fit_psmp_dtd_rbst,~,~] = nt_detrend(psmp(~ind_nan),cfg.polyorder);
                
                iptr  = 1;
                for is = 1:length(ind_nan)
                    if ind_nan(is) == 1

                    else
                        pupil_detrend_rbst(is) = fit_psmp_dtd_rbst(iptr);
                        iptr = iptr + 1;
                    end
                end
            
            % Find the indices of events 
            % idx ( : , 1/STIM 2/RESP 3/FBCK 4/END)
            [~,imsg] = pupil_event_indexer_rlvsl(data_eye);
            imsg = imsg(3+4*(0:nt-1));
            
            % Extract epochs
            for it = 1:nt
                if excluded_trials(it,ctr_blck,ctr_subj)
                    continue
                end
                ctr_epoch = ctr_epoch+1;
                tstart = imsg(it) - epoch_window(1);
                tend   = imsg(it) + epoch_window(2);
                epochs(ctr_epoch,:) = pupil_detrend_rbst(tstart:tend); % log the epoch
                idx_subj_epoch(ctr_epoch) = ctr_subj;    % log the relative subj number
            end
        end
    end
end

excluded_trials = excluded_trials(:);
fbs = fbs(:);
fbs = fbs(~excluded_trials);
qts = qts(:);
qts = qts(~excluded_trials);

ind_epoch_nan = false(size(epochs,1),1); % indicates epochs where NaN is found

out = struct;


if cfg.incl_nan
    out.epochs         = epochs;
    out.fbs            = fbs;
    out.idx_subj       = idx_subj_epoch;
    out.qts            = qts;
else
    % Identify epochs where NaNs are found and omit them in output
    for ie = 1:size(epochs,1) % go through each epoch
        if sum(isnan(epochs(ie,:)))>0
            ind_epoch_nan(ie) = 1;
        end
    end
    out.epochs         = epochs(~ind_epoch_nan,:);
    out.fbs            = fbs(~ind_epoch_nan);
    out.idx_subj       = idx_subj_epoch(~ind_epoch_nan,:);
    out.qts            = qts(~ind_epoch_nan);
end
out.epoch_window = epoch_window;

end