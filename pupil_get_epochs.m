function [out] = pupil_get_epochs(subjlist,cfg)
% pupil_get_epochs
%
% Usage: Organizes and detrends preprocessed pupil data into epochs for all included 
%           subjects to be analyzed for experiment RLVSL.
%
% Input:        subjlist - array of subjects to be included in analysis
%               setrange - hardcoded right epoch window limit (seconds)
%               cfg      - structure containing fields:
%    (required)             ievent    : onset event of interest
%    (optional)             ievent_end: ending event of interest
%    (optional)             lim_epoch : right-most limit of an epoch (STIM or END)
%    (optional)             min_prefb : minimum sampling points a trial must have before
%                                        feedback onset      
%    (optional)             incl_nan  : option to include epochs where NaN were found
%    (optional)             polyorder : order of polynomial for detrending algorithm
%    (optional)             isdetrend : flag for detrending the pupil data
%    (optional)             iszscored : z-score the pupil data within given subject
%
% Output: out - structure containing fields:
%                   fbs      : feedback values
%                   epochs   : pupil data within epochs
%                   idx_subj : subject id for fb and pupil trends 
%                               (relative id, not absolute!)
%                   
% Jun Seok Lee <jlexternal@gmail.com>

if nargin < 2
    cfg = [];
end
if nargin < 2
    error('Provide condition to analyze over!');
end
if nargin < 1
    error('Missing subject list and experimental condition!');
end
if ~isfield(cfg,'r_ep_lim')
    cfg.r_ep_lim = 0;
end
if ~isfield(cfg,'incl_nan')
    cfg.incl_nan = false;
end
if ~isfield(cfg,'lim_epoch')
    cfg.lim_epoch = 'STIM';
end
if ~isfield(cfg,'isdetrend')
    cfg.isdetrend = true;
end
if ~isfield(cfg,'iszscored')
    cfg.iszscored = false;
end
if ~isfield(cfg,'ievent')
    error('Indicate which event (1/STIM 2/RESP 3/FBCK 4/END) is considered the onset event.');
end
ievent = cfg.ievent; % onset event : 1/STIM 2/RESP 3/FBCK 4/END
if ~isfield(cfg,'ievent_end')
    ievent_end = 4;
else
    if ~ismember(cfg.ievent_end,1:4)
        error('Invalid ending event index provided!');
    else
        ievent_end = cfg.ievent_end;
    end
end


% Add robust detrending toolbox path
addpath('./Toolboxes/NoiseTools/');

% Find the epoch window that fits all subjects
[epoch_window,excluded_trials,fbs,rsp,qts,trs,bks,cds] = pupil_epoch_window_rlvsl(subjlist,cfg);

if cfg.r_ep_lim ~= 0 
    epoch_window(2) = cfg.r_ep_lim*500; % 500 samples per second
end

fbs = fbs-50; % make the negative feedback actually negative
nt = size(fbs,1);
nb = size(fbs,2);
n_excld             = sum(sum(sum(excluded_trials)));
epochs              = nan(numel(fbs)-n_excld,sum(epoch_window)+1); % pupil size values within a given epoch
idx_range_interest  = false(size(epochs)); % logical indexing array of epoch samples within interest range
idx_subj_epoch      = zeros(numel(fbs)-n_excld,1);

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
        if strcmpi(expe(ib).type,'training')
            continue
        end
        ctr_blck = ctr_blck+1;

        % Load the pupil data
        pupilfile = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b%02d*preproc.mat',isubj,isubj,ib));
        load(sprintf('./Data/S%02d/%s',isubj,pupilfile.name)); % loads structure 'data_eye' to workspace
        tsmp = data_eye.tsmp;
        psmp = data_eye.psmp;
        tmsg = data_eye.tmsg;

        % Normalize the time sample points
        t_init = tsmp(1);
        tsmp_normd = (tsmp-t_init)/1000;

        % Identify NaN in data
        ind_nan = isnan(tsmp_normd);
        ind_nan = ind_nan | isnan(psmp);

        % Detrend the data
        if cfg.isdetrend
            % Detrend data with Robust Detrending (of Alain de Cheveigné)
            pupil_detrend_rbst = nan(size(psmp));
            [fit_psmp_dtd_rbst,~,~] = nt_detrend(psmp(~ind_nan),cfg.polyorder);

            % Log non-NaN detrended data
            iptr  = 1;
            for is = 1:length(ind_nan)
                if ind_nan(is) == 1

                else
                    pupil_detrend_rbst(is) = fit_psmp_dtd_rbst(iptr);
                    iptr = iptr + 1;
                end
            end
        end

        % Find the indices of events 
        % idx ( : , 1/STIM 2/RESP 3/FBCK 4/END)
        [~,imsg] = pupil_event_indexer_rlvsl(data_eye);
        
        imsg_end = imsg(ievent_end+4*(0:nt-1));
        imsg     = imsg(ievent+4*(0:nt-1));
        
        % Extract epochs
        for it = 1:nt
            tstart = imsg(it) - epoch_window(1);
            tend   = imsg(it) + epoch_window(2);
            
            if excluded_trials(it,ctr_blck,ctr_subj)
                if ~cfg.incl_nan
                    continue
                else
                    ctr_epoch = ctr_epoch+1;
                    epochs(ctr_epoch,:) = nan(1,length(tstart:tend)); % log the epoch
                end
            else
                ctr_epoch = ctr_epoch+1;
                if cfg.isdetrend
                    epochs(ctr_epoch,:) = pupil_detrend_rbst(tstart:tend);  % log detrended data
                else
                    epochs(ctr_epoch,:) = psmp(tstart:tend);                % log raw data
                end
            end
            idx_range_interest(ctr_epoch,epoch_window(1):epoch_window(1)+imsg_end(it)-imsg(it)) = true; % log the locations of pupil samples within interest range
            idx_subj_epoch(ctr_epoch) = ctr_subj;    % log the relative subj number
        end
    end
    if cfg.iszscored
        zscore_xnan = @(x) bsxfun(@rdivide, bsxfun(@minus, x, mean(x,'all','omitnan')), std(x,1,'all','omitnan'));
        epochs_z = zscore_xnan(epochs(idx_subj_epoch == ctr_subj,:));
        epochs(idx_subj_epoch == ctr_subj,:) = epochs_z;
    end
end
idx_range_interest = idx_range_interest(:,1:size(epochs,2));


excluded_trials = excluded_trials(:);
fbs = fbs(:); 
rsp = rsp(:);
qts = qts(:); 
trs = trs(:); 
bks = bks(:); 
cds = cds(:);

ind_epoch_nan = false(size(epochs,1),1); % indicates epochs where NaN is found

% Identify epochs where NaNs are found
for ie = 1:size(epochs,1) % go through each epoch
    if sum(isnan(epochs(ie,:)))>0
        ind_epoch_nan(ie) = 1;
    end
end
out = struct;


if cfg.incl_nan
    out.epochs         = epochs;
    out.fbs            = fbs;
    out.rsp            = rsp;
    out.idx_subj       = idx_subj_epoch;
    out.qts            = qts;
    out.bks            = bks;
    out.trs            = trs;
    out.cds            = cds;
    out.ind_epoch_nan  = ind_epoch_nan;
    out.idx_range_interest = idx_range_interest;
else
    fbs = fbs(~excluded_trials);
    rsp = rsp(~excluded_trials);
    qts = qts(~excluded_trials);
    trs = trs(~excluded_trials);
    bks = bks(~excluded_trials);
    
    out.epochs         = epochs(~ind_epoch_nan,:);
    out.fbs            = fbs(~ind_epoch_nan);
    out.rsp            = rsp(~ind_epoch_nan);
    out.idx_subj       = idx_subj_epoch(~ind_epoch_nan,:);
    out.qts            = qts(~ind_epoch_nan);
    out.bks            = bks(~ind_epoch_nan);
    out.trs            = trs(~ind_epoch_nan);
    out.cds            = cds(~ind_epoch_nan);
    out.idx_range_interest = idx_range_interest;
end
out.epoch_window = epoch_window;

end