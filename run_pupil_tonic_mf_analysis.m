% run_pupil_analysis.m
% 
% Usage: Tonic pupil analysis (pre-detrending analysis)

clear all;
addpath('./Toolboxes/'); % add path containing ANOVA function
addpath('./functions_local')
%% Epoch raw data (w/o detrend)

% subjects to be included in analysis
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

epc_struct = {};

% Check pupil_get_epochs.m for optional config.

cfg = struct; 
%cfg.polyorder = 20; % default is nt
cfg.r_ep_lim    = 4; % set the rightward limit of epoch window (in seconds)
cfg.incl_nan    = true;
cfg.isdetrend   = false;
cfg.iszscored   = true;
cfg.lim_epoch   = 'END';
cfg.ievent      = 1; % 1/STIM 2/RESP 3/FBCK 4/END
cfg.ievent_end  = 4;

epc_struct = pupil_get_epochs(subjlist,cfg);

% choose the smallest epoch window for comparison
epc_range    = min([epc_struct.epoch_window],[],1); % [nSamples before onset, nSamples after onset]
epc_fb_onset = epc_struct.epoch_window(1) +1;

ifig = 0;

%% 1a. Organize: Tonic trial-wide pupil dilations (z-scored) (STIM to END)

% This analysis looks at the average pupil dilation (z-scored) from the onset of the
% stimulus to the end of the trial across the 16 trials in a block for all subjects
% in all conditions, grouped by whether the subject was biased or unbiased in a given
% quarter. 

% Load epsilon-greedy percentages
load('out_fit_epsi.mat'); 
epsi(:,:,:,1) = []; % [subj,condition,quarter]
epsi(epsi<=.5) = 0;
epsi(epsi~=0) = 1;

nt = 16;
pupil_means_trial = nan(nsubj,nt,2); % [subj,trial,bias]

epochs  = epc_struct.epochs;
idx_subj= epc_struct.idx_subj;
trs     = epc_struct.trs;
qts     = epc_struct.qts;
cds     = epc_struct.cds;
intrange= epc_struct.idx_range_interest;

for isubj = 1:nsubj
    
    % Identify biased conditions and quarters for the subject
    biased_c_q = []; %[condition, quarter]
    for ic = 1:3
        for iq = 1:4
            if epsi(isubj,ic,iq) == 1
                biased_c_q = cat(1,biased_c_q,[ic iq]);
            end
        end
    end
    
    idx_c_q = false(size(trs));
    for ibias = 1:2 % 1/unbiased; 2/biased
        
        % Determine logical array of (un)biased quarters and conditions
        if ibias == 1
            % Build up logical array of biased quarters
            for i = 1:size(biased_c_q,1)
                idx_c_q = bsxfun(@or,idx_c_q,bsxfun(@and,cds==biased_c_q(i,1),qts==biased_c_q(i,2)));
            end
            idx_c_q = ~idx_c_q; % turn them into unbiased quarters
        else
            if isempty(biased_c_q) % account for subjects who are never biased
                continue 
            end
            idx_c_q = ~idx_c_q; % turn them into biased quarters
        end
        
        for it = 1:nt
            epoch_it = epochs(idx_subj==isubj & trs==it & idx_c_q,:);       % identify all epochs of trial it
            intrange_it = intrange(idx_subj==isubj & trs==it & idx_c_q,:);  % single out samples within interest range

            pupil_means_trial(isubj,it,ibias) = mean(mean(epoch_it(intrange_it),2,'omitnan'),'omitnan');
        end
    end
    
end

% Count number of subjects who are never biased (for SEM calculation)
nonbiased_subjs = [];
for i = 1:4
    nonbiased_subjs = cat(2,nonbiased_subjs,epsi(:,:,i));
end
nonbiased_subjs = numel(find(sum(nonbiased_subjs,2) == 0));

% Statistics
h = zeros(1,nt);
p = h;
for it = 1:16
    [h(it),p(it)] = ttest2(pupil_means_trial(:,it,1),pupil_means_trial(:,it,2),'VarType','unequal'); % Welch's t-test (for unequal sample size)
end

%% 1b. Plot: Trial-wide pupil dilation (z-scored) grouped on BIASED vs UNBIASED blocks
figure;
hold on;
plotrgb = [.5 .5 .5;.8 .2 .2];
for ibias = 1:2
    if ibias == 2
        nsubj_temp = nsubj-nonbiased_subjs;
    else
        nsubj_temp = nsubj;
    end
    shadedErrorBar(1:nt,mean(pupil_means_trial(:,:,ibias),1,'omitnan'),std(pupil_means_trial(:,:,ibias),0,1,'omitnan')/sqrt(nsubj_temp),...
        'lineprops',{'Color',plotrgb(ibias,:),'LineWidth',2},'patchSaturation',0.075);
    for it = 1:nt
        scatter(it,mean(pupil_means_trial(:,it,ibias),1,'omitnan'),60,'MarkerFaceColor',plotrgb(ibias,:),'MarkerFaceAlpha',.8,...
            'MarkerEdgeColor',[0 0 0],'MarkerEdgeAlpha',h(it),'LineWidth',1.5,'HandleVisibility','off');
    end
end
xticks([4 8 12 16]);
ylabel('pupil dilation (z)','FontSize',12);
legtxt = {'unbiased','biased'};
legend(legtxt,'Location','northeast');

%% 2a. Organize: Pupil dilation on SWITCH trials accounting for general trial-wide trends

% This analysis looks at the average pupil dilation (z-scored) around switch trials, having
% subtracted general trends from the above analysis (1a). Switch trials that occur
% within the window of the main switch trial are disregarded.

% Note: Code section 1a must be run previous to this.

rsp = epc_struct.rsp;
bks = epc_struct.bks;

% messy stuff to get relative and absolute indexing of blocks within the experiment
% or condition
nc = 3;
nb_c = 16;
resps = nan(nb_c,nt,3,nsubj);
nb = 48;
ntrain = 3;
for isubj = subjlist
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj);
    load(filename,'expe');
    ib_c = ones(3,1);
    for ib = 1+ntrain:nb+ntrain
        ctype = expe(ib).type;
        switch ctype
            case 'rnd' % random across successive blocks
                icond = 3;
            case 'alt' % always alternating
                icond = 2;
            case 'rep' % always the same
                icond = 1;
        end
        resps(ib_c(icond),:,icond,isubj)    = expe(ib).resp;
        ib_c(icond) = ib_c(icond)+1;
    end
end

nt_bfr = 4; % number of trials around switch point (buffer trials)
pupil_switch_means = nan(nsubj,nt_bfr*2+1);

for isubj = 1:nsubj
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',subjlist(isubj),subjlist(isubj));
    load(filename,'expe');
    
    bk_abs = nan(nb_c,nc);
    bk_ctr = ones(1,nc);
    for ib = 4:length(expe)
        ctype = expe(ib).type;
        switch ctype
            case 'rnd' % random across successive blocks
                icond = 3;
            case 'alt' % always alternating
                icond = 2;
            case 'rep' % always the same
                icond = 1;
        end
        bk_abs(bk_ctr(icond),icond) = ib-3;
        bk_ctr(icond) = bk_ctr(icond) + 1;
    end
    
    resps_s = resps(:,:,:,subjlist(isubj));
    
    epochs_switch = [];
    
    for ic = 1:nc
        for ib = 1:nb_c
            % find absolute block number
            ib_abs = bk_abs(ib,ic); 
            
            % calculate quarter of block
            iq = ceil((ib-eps)/4);
            % determine whether quarter was biased or not
            if epsi(isubj,ic,iq) == 0
                biasflag = false;
            else
                biasflag = true;
            end
            
            % find switch trials
            for it = 2:nt
                if resps_s(ib,it,ic) ~= resps_s(ib,it-1,ic)
                    % NaNs for left side of switch
                    if it-nt_bfr <= 0
                        bfr_l   = nt_bfr-it+1;  % left NaN buffer 
                        range_l = 1;            % left range limit
                    else
                        bfr_l   = 0;
                        range_l = it-nt_bfr;
                    end
                    % NaNs for right side of switch
                    if nt-it-nt_bfr < 0
                        bfr_r   = -(nt-it-nt_bfr);  % right NaN buffer
                        range_r = nt;               % right range limit
                    else
                        bfr_r   = 0;
                        range_r = it+nt_bfr;
                    end
                    
                    % Account for switches that happen within the range limits and
                    %   mark them for exclusion
                    excl_tr = [];
                    for jt = 1:4
                        % Check if pointer at the left end of the response array
                        if it-jt > 1
                            % Find switches before main switch and mark
                            if resps_s(ib,it-jt,ic) ~= resps_s(ib,it-jt-1,ic)
                                excl_tr = cat(2,excl_tr,nt_bfr+1-jt);
                            end
                        end
                        % Check if pointer at the right end of the response array
                        if it+jt <= 16
                            % Find switches before main switch and mark
                            if resps_s(ib,it+jt,ic) ~= resps_s(ib,it+jt-1,ic)
                                excl_tr = cat(2,excl_tr,nt_bfr+1+jt);
                            end
                        end
                    end
                    
                    % Isolate pupil means around the switch window
                    pups = mean(epochs(idx_subj==isubj & ismember(trs,range_l:range_r) & cds==ic & bks == ib_abs,:),2,'omitnan')';
                    
                    % Subtract general effect of RTs based on trial position and bias
                    if biasflag
                        pupil_means_normd = bsxfun(@minus,...
                            [nan(1,bfr_l) pups nan(1,bfr_r)], ...
                            [nan(1,bfr_l) pupil_means_trial(isubj,range_l:range_r,2) nan(1,bfr_r)]);
                    else
                        pupil_means_normd = bsxfun(@minus,...
                            [nan(1,bfr_l) pups nan(1,bfr_r)], ...
                            [nan(1,bfr_l) pupil_means_trial(isubj,range_l:range_r,1) nan(1,bfr_r)]);
                    end
                    pupil_means_normd(excl_tr) = NaN;
                    epochs_switch = cat(1,epochs_switch,pupil_means_normd);
                end
            end
        end
    end
    pupil_switch_means(isubj,:) = mean(epochs_switch,1,'omitnan');
end

% Statistics
h = zeros(1,size(pupil_switch_means,2));
p = h;
for it = 1:size(pupil_switch_means,2)
    [h(it),p(it)] = ttest(pupil_switch_means(:,it)); 
end

%% 2b. Plot: Pupil dilation on SWITCH trials accounting for general trial-wide trends
figure;
plotrgb = [.5 .5 .8];
hold on;
for ibias = 1:2
    shadedErrorBar(1:size(pupil_switch_means,2),mean(pupil_switch_means,1,'omitnan'),std(pupil_switch_means,0,1,'omitnan')/sqrt(nsubj),...
        'lineprops',{'Color',plotrgb,'LineWidth',2},'patchSaturation',0.075);
    for it = 1:size(pupil_switch_means,2)
        scatter(it,mean(pupil_switch_means(:,it),1,'omitnan'),60,'MarkerFaceAlpha',.8,'MarkerFaceColor',plotrgb,...
            'MarkerEdgeColor',[0 0 0],'MarkerEdgeAlpha',h(it),'LineWidth',1.5,'HandleVisibility','off');
    end
end
xticklabels(-4:4);
ylabel('pupil dilation (z)','FontSize',12);
yline(0);
xline(5,'--');

%% Local functions
function rgb = graded_rgb(ic,iq,nq)
    xc = linspace(.8,.2,nq);

    rgb =  [1,xc(iq),xc(iq); ...
               xc(iq),1,xc(iq); ...
               xc(iq),xc(iq),1];

    rgb = rgb(ic,:);
end
