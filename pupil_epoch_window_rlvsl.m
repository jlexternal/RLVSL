function [epoch_window,excluded_trials,fbs,rsp,qts,trs,bks] = pupil_epoch_window_rlvsl(subjlist,condtype,cfg)
% pupil_epoch_window_rlvsl
%
% Usage: Finds the smallest epoch window around a feedback point on all trials of all
% subjects
%
% Input:    subjlist - list of subjects to analyze
%           condtype - condition to analyze over
%           cfg      - structure containing fields:
%                           lim_epoch : 'STIM' or 'END'
%                                      % STIM : nt-1 trials analyzed
%                                      % END  : all nt trials analyzed
%                           min_prefb : minimum sampling points a trial must have before fb
%                                       to be considered
%
% Output:   epoch_window - indices to be subtracted from and added to the feedback
%                           index for proper epoch window
%
% Jun Seok Lee <jlexternal@gmail.com>

if nargin < 1
    error('Missing subject list and experimental condition!');
end
if nargin < 2
    error('Provide condition to analyze over!');
end
if ~isfield(cfg,'lim_epoch')
    cfg.lim_epoch = 'STIM';
end
if ~isfield(cfg,'min_prefb')
    cfg.min_prefb = 250; % 250*2ms = 500ms = .5s
end
min_prefb = cfg.min_prefb;
lim_epoch = cfg.lim_epoch;
if ~ismember(condtype,{'rep','alt','rnd'})
    error('Unrecognisable condition!');
end

epoch_window = [1e9 1e9];

ctr_excl = 0;
ctr_subj = 0;
subj_init = true;
for isubj = subjlist
    load(sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)); % loads structure 'expe' to workspace
    ctr_subj = ctr_subj+1;
    if subj_init
        nt   = expe(1).cfg.ntrls;
        incr_right = 1; % if right limit is END1
        if strcmpi(lim_epoch,'STIM')
            nt = nt-1; % if the right limit is the next stimulus, only consider the first nt-1 trials
            incr_right = incr_right+1;
        end
        % set constants and allocate memory
        nb = length(expe);
        excluded_trials  = false(nt,(nb-3)/3,numel(subjlist)); % logical matrix of excluded trials
        fbs = nan(nt,(nb-3)/3,numel(subjlist)); % matrix of feedback values
        rsp = nan(size(fbs));   % subject responses
        qts = nan(size(fbs));   % matrix of quarters
        mgen = expe(1).cfg.mgen; % generative mean
        sgen = expe(1).cfg.sgen; % generative std
        idx_fb = 3+4*(0:nt-1);  % indices of feedback given it is the 3rd out of every 4 events
        subj_init = false;
    end
    
    ctr_blck = 0;
    for ib = 1:nb
        if strcmpi(expe(ib).type,condtype)
            ctr_blck = ctr_blck+1;
            % Load the pupil data
            pupilfile = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b%02d*preproc.mat',isubj,isubj,ib));
            load(sprintf('./Data/S%02d/%s',isubj,pupilfile.name)); % loads structure 'data_eye' to workspace
            tmsg      = data_eye.tmsg;
            
            diffs = zeros(2,nt);
            % Identify potential epochs
            for it = 1:nt
                i_fb = idx_fb(it);
                diffs(1,it) = abs(tmsg(i_fb) - tmsg(i_fb-2));          % |FBCK1 - STIM1| 
                diffs(2,it) = abs(tmsg(i_fb) - tmsg(i_fb+incr_right)); % |FBCK1 - successive event (END1 or STIM2)|  
                fbs(it,ctr_blck,ctr_subj) = convert_fb_raw2seen(expe(ib).blck(it),expe(ib).resp(it),mgen,sgen); % requires function in path
                rsp(it,ctr_blck,ctr_subj) = expe(ib).resp(it);
                qts(it,ctr_blck,ctr_subj) = floor((expe(ib).sesh-1)/2)+1;
                
                % Exclude trials where the subject responded too fast and would not
                % have seen the stimulus i.e. the time between the feedback and
                % stimulus is much too short for a true reaction to have happened
                if diffs(1,it) < min_prefb
                    excluded_trials(it,ctr_blck,ctr_subj) = true;
                    ctr_excl = ctr_excl+1;
                    diffs(1,it) = 1e9; 
                end
            end
            min_diff = [min(diffs(1,:)) min(diffs(2,:))];
            epoch_window = min(epoch_window,min_diff);
        else
            continue
        end
    end
end

% output trial and block indices
trs = repmat((1:nt)',[1 (nb-3)/3 numel(subjlist)]);
bks = repmat((1:(nb-3)/3),[nt 1 numel(subjlist)]);

%fprintf('%d trials counted as excluded from analysis\n',ctr_excl);
%fprintf('%d trials indicated as excluded in output exclusion matrix\n',sum(sum(sum(excluded_trials))));
if ctr_excl ~= sum(sum(sum(excluded_trials)))
    error('Excluded trials counted do not match those logged in exclusion matrix! Check code!');
end
end