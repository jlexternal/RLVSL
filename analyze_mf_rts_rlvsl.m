% Model-free analysis of behavior on experiment RLVSL
%
clear all;
close all;
addpath('./Toolboxes');
condtypes = ["Repeating","Alternating","Random"];
%% Load data
ifig = 1;
nsubjtot    = 31;
excluded    = [1];
subjlist    = setdiff(1:nsubjtot, excluded);
subparsubjs = [excluded 23 28];
subjlist = setdiff(1:nsubjtot, subparsubjs); % if excluding underperforming/people who didn't get it
% load experiment structure
nsubj = numel(subjlist);
% Data manip step
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',2,2);
load(filename,'expe');
nt = expe(1).cfg.ntrls;
nb = expe(1).cfg.nbout;
ntrain = expe(1).cfg.ntrain;
nc = 3;
nb_c = nb/nc;

blcks       = nan(nb_c,nt,3,nsubj);
resps       = nan(nb_c,nt,3,nsubj);
rts         = nan(nb_c,nt,3,nsubj);

mu_new   = 55;  % mean of higher distribution
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

a = sig_opti/expe(1).cfg.sgen;      % slope of linear transformation aX+b
b = mu_new - a*expe(1).cfg.mgen;    % intercept of linear transf. aX+b

run_epiphany_qts; % get epiphany quarters

for isubj = subjlist
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj);
    fprintf('Loading subject %d...\n',isubj);
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

        resp_mult = -(expe(ib).resp-1.5)*2;
        blcks(ib_c(icond),:,icond,isubj) = round(resp_mult.*expe(ib).blck*a+b);
        
        rts(ib_c(icond),:,icond,isubj)      = expe(ib).rt;
        resps(ib_c(icond),:,icond,isubj)    = expe(ib).resp;
        ib_c(icond) = ib_c(icond)+1;
    end
end
blcks = (blcks-50)/100;

%% 1a. Organize: Correlation between feedback magnitude and RT

coefs_rt_fb = zeros(nsubj,4,nc); 
pvals = zeros(nc,4);

for icond = 1:nc
    for iq = 1:4
        for isubj = 1:nsubj
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            rts_vec = rts(blockrange,2:nt,icond,subjlist(isubj)); % rts on trial t
            rts_vec = rts_vec(:);
            rsp_vec = blcks(blockrange,1:nt-1,icond,subjlist(isubj)); % fb on trial t-1
            rsp_vec = rsp_vec(:);
            bval = regress(rts_vec,[ones(size(rsp_vec)) rsp_vec]);
            coefs_rt_fb(isubj,iq,icond) = bval(2);
            
        end 
        [~,pvals(icond,iq)] = ttest(coefs_rt_fb(:,iq,icond));
    end
end

%% 1b. Plot: Correlation between feedback magnitude and RT
figure;
hold on;
handleVis = 'off';
for icond = 1:nc
    for iq = 1:4
        if iq == 4
            handleVis = 'on';
        end
        scatter(iq,mean(coefs_rt_fb(:,iq,icond)),'filled','MarkerEdgeColor',graded_rgb(icond,iq,4),'MarkerFaceColor',graded_rgb(icond,iq,4),...
                'HandleVisibility',handleVis);
        errorbar(iq,mean(coefs_rt_fb(:,iq,icond)),std(coefs_rt_fb(:,iq,icond))/sqrt(nsubj),'Color',graded_rgb(icond,iq,4),...
                'HandleVisibility','off');
        text(iq+.1,mean(coefs_rt_fb(:,iq,icond)),sigstar(pvals(icond,iq)));
    end
    handleVis = 'off';
end
XTick = 1:4;
set(gca,'xtick',XTick);
yline(0,':');
ylabel('Regression weight beta','FontSize',14);
xlabel('Quarter','FontSize',14);
xlim([.5 4.5]);
legend(condtypes,'Location','southeast');
title(sprintf('Effect of feedback value on reaction times\nError bars: SEM'),'FontSize',12);

%% 2a. Organize: Correlation between feedback magnitude and RT before and after epiphanies within conditions
coefs_rt_fb_epiph_rep = zeros(nsubj,2); 
coefs_rt_fb_epiph_alt = zeros(nsubj,2); 
coefs_rt_fb_epiph_r_rnd   = zeros(nsubj,2);
coefs_rt_fb_epiph_a_rnd   = zeros(nsubj,2);

pvals_epiph = zeros(2,2);
pvals_epiph_rnd = zeros(2,2);

pre_ep_qts = 0;
for icond = 1:2
    isubj = 0;
    excl_epiph = [];
    for subj = subjlist
        isubj = isubj + 1;
        % pre and post epiphany analysis
        if icond == 1
            if ~isempty(prepost_epiph_rep{subj,1})
                pre_ep_qts = prepost_epiph_rep{subj,1};
            else
                excl_epiph = [excl_epiph isubj];
                continue
            end
        elseif icond == 2
            if ~isempty(prepost_epiph_alt{subj,1})
                pre_ep_qts = prepost_epiph_alt{subj,1};
            else 
                excl_epiph = [excl_epiph isubj];
                continue
            end
        end

        % Regression analysis
        blockrange= [];
        for iq = pre_ep_qts
            blockrange = [blockrange 4*(iq-1)+1:4*(iq-1)+4];
        end
        for iepiph = 1:2
            if iepiph == 2 % after epiphany
                blockrange = setdiff(1:16,blockrange);
            end
            rts_vec = rts(blockrange,2:nt,icond,subj); % rts on trial t
            rts_vec = rts_vec(:);
            rsp_vec = blcks(blockrange,1:nt-1,icond,subj); % fb on trial t-1
            rsp_vec = rsp_vec(:);
            bval_epiph = regress(rts_vec,[ones(size(rsp_vec)) rsp_vec]);
            if icond == 1
                coefs_rt_fb_epiph_rep(isubj,iepiph) = bval_epiph(2);
            else
                coefs_rt_fb_epiph_alt(isubj,iepiph) = bval_epiph(2);
            end

            % Regress on the RND condition with the same quarters as control
            rts_vec = rts(blockrange,2:nt,3,subj); % rts on trial t
            rts_vec = rts_vec(:);
            rsp_vec = blcks(blockrange,1:nt-1,3,subj); % fb on trial t-1
            rsp_vec = rsp_vec(:);
            bval_epiph = regress(rts_vec,[ones(size(rsp_vec)) rsp_vec]);
            if icond == 1
                coefs_rt_fb_epiph_r_rnd(isubj,iepiph) = bval_epiph(2);
            else
                coefs_rt_fb_epiph_a_rnd(isubj,iepiph) = bval_epiph(2);
            end
        end
    end
    
    if icond == 1
        coefs_rt_fb_epiph_rep(excl_epiph,:,:)   = [];
        coefs_rt_fb_epiph_r_rnd(excl_epiph,:,:) = [];
        [~,pvals_epiph(icond,1)] = ttest(coefs_rt_fb_epiph_rep(:,1));
        [~,pvals_epiph(icond,2)] = ttest(coefs_rt_fb_epiph_rep(:,2));
        [~,pvals_epiph_rnd(icond,1)] = ttest(coefs_rt_fb_epiph_r_rnd(:,1));
        [~,pvals_epiph_rnd(icond,2)] = ttest(coefs_rt_fb_epiph_r_rnd(:,2));
    else
        coefs_rt_fb_epiph_alt(excl_epiph,:,:)   = [];
        coefs_rt_fb_epiph_a_rnd(excl_epiph,:,:) = [];
        [~,pvals_epiph(icond,1)] = ttest(coefs_rt_fb_epiph_alt(:,1));
        [~,pvals_epiph(icond,2)] = ttest(coefs_rt_fb_epiph_alt(:,2));
        [~,pvals_epiph_rnd(icond,1)] = ttest(coefs_rt_fb_epiph_a_rnd(:,1));
        [~,pvals_epiph_rnd(icond,2)] = ttest(coefs_rt_fb_epiph_a_rnd(:,2));
    end
    
    pre_ep_qts = 0;
end

%% 2b. Statistics: Correlation between feedback magnitude and RT before and after epiphanies within conditions
% 
epiph_data_rep = cat(3,coefs_rt_fb_epiph_rep,coefs_rt_fb_epiph_r_rnd);
epiph_data_alt = cat(3,coefs_rt_fb_epiph_alt,coefs_rt_fb_epiph_a_rnd);

tbl_epiph_rep = simple_mixed_anova(epiph_data_rep,[],{'Epiphany','Condition'});
tbl_epiph_alt = simple_mixed_anova(epiph_data_alt,[],{'Epiphany','Condition'});

%% 2c. Plot: Correlation between feedback magnitude and RT before and after epiphanies within conditions
figure;
hold on;
handleVis = 'off';
mrkshape = '';
for icond = 1:2
    if icond == 2
        mrkshape = '^';
    end
    for iepiph = 1:2
        if iepiph == 2
            handleVis = 'on';
        else
            handleVis = 'off';
        end
        
        if icond == 1
            coefs_rt_fb_epiph = coefs_rt_fb_epiph_rep(:,iepiph);
            coefs_rt_fb_epiph_rnd = coefs_rt_fb_epiph_r_rnd(:,iepiph);
        else
            coefs_rt_fb_epiph = coefs_rt_fb_epiph_alt(:,iepiph);
            coefs_rt_fb_epiph_rnd = coefs_rt_fb_epiph_a_rnd(:,iepiph);
        end
        
        % REP and ALT conditions
        errorbar(iepiph+(icond-1.5)*.2,mean(coefs_rt_fb_epiph),std(coefs_rt_fb_epiph)/sqrt(nsubj),'Color',graded_rgb(icond,iepiph,2),...
                'HandleVisibility','off');
        scatter(iepiph+(icond-1.5)*.2,mean(coefs_rt_fb_epiph),50,mrkshape,'filled','MarkerEdgeColor',graded_rgb(icond,iepiph,4),'MarkerFaceColor',graded_rgb(icond,iepiph,2),...
                'HandleVisibility',handleVis);
%        text(iepiph+(icond-1.5)*.4,mean(coefs_rt_fb_epiph),sigstar(pvals_epiph(icond,iepiph)),'HorizontalAlignment','center');
        
        % RND conditions relative to the blocks
        errorbar(iepiph+(icond-1.5)*.2,mean(coefs_rt_fb_epiph_rnd),std(coefs_rt_fb_epiph_rnd)/sqrt(nsubj),'Color',graded_rgb(3,iepiph,2),...
                'HandleVisibility','off');
        scatter(iepiph+(icond-1.5)*.2,mean(coefs_rt_fb_epiph_rnd),50,mrkshape,'filled','MarkerEdgeColor',graded_rgb(3,iepiph,4),'MarkerFaceColor',graded_rgb(3,iepiph,2),...
                'HandleVisibility',handleVis);
%        text(iepiph+(icond-1.5)*.4,mean(coefs_rt_fb_epiph_rnd),sigstar(pvals_epiph_rnd(icond,iepiph)),'HorizontalAlignment','center');
    end
end

yline(0,':');
ylabel('Regression weight beta','FontSize',14);
xlabel('Quarters; circum-epiphany','FontSize',14);
xticks([1 2]);
xticklabels({'Pre', 'Post'});
xlim([.5 2.5]);
ylim([-.9 .1]);
legend({'Repeating','Random (locked to Rep.)','Alternating','Random (locked to Alt.)'},'Location','southeast');
title(sprintf('Effect of feedback value on reaction times\n(%d/%d subjects) Error bars: SEM',size(coefs_rt_fb_epiph_rnd,1),nsubj),'FontSize',12);

%% 3a. Organize: Correlation of RTs and feedback magnitude grouped on BIASED vs UNBIASED blocks

% Load epsilon-greedy percentages
load('out_fit_epsi.mat');
epsi(:,:,:,1) = [];
epsi(epsi<=.5) = 0;
epsi(epsi~=0) = 1;

bvals = zeros(nsubj,2); %isubj,ibias

for isubj = 1:nsubj
    rts_vec1 = [];
    rts_vec2 = [];
    rsp_vec1 = [];
    rsp_vec2 = [];
    for icond = 1:nc
        for iq = 1:4
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            if epsi(isubj,icond,iq) == 0
                rts_vec1 = cat(1,rts_vec1,rts(blockrange,2:nt,icond,subjlist(isubj)));
                rsp_vec1 = cat(1,rsp_vec1,blcks(blockrange,1:nt-1,icond,subjlist(isubj)));
            else
                rts_vec2 = cat(1,rts_vec2,rts(blockrange,2:nt,icond,subjlist(isubj)));
                rsp_vec2 = cat(1,rsp_vec2,blcks(blockrange,1:nt-1,icond,subjlist(isubj)));
            end
        end 
    end
    
    % Regress
    for ibias = 1:2
        if ibias == 1
            rts_vec = rts_vec1(:);
            rsp_vec = rsp_vec1(:);
        else
            rts_vec = rts_vec2(:);
            rsp_vec = rsp_vec2(:);
        end
        bval = regress(rts_vec,[ones(size(rsp_vec)) rsp_vec]);
        bvals(isubj,ibias) = bval(2);
    end
end
bvals(bvals == 0) = nan;
nonbiased_subjs = [];
for i = 1:4
    nonbiased_subjs = cat(2,nonbiased_subjs,epsi(:,:,i));
end
nonbiased_subjs = numel(find(sum(nonbiased_subjs,2) == 0));

[h,p] = ttest2(bvals(:,1),bvals(:,2),'VarType','unequal'); % Welch's t-test (for unequal sample size)

%% 3b. Plot: Correlation of RTs and feedback magnitude grouped on BIASED vs UNBIASED blocks
figure;
hold on;
for ibias = 1:2
    if ibias == 2
        nsubj_temp = nsubj-nonbiased_subjs;
    else
        nsubj_temp = nsubj;
    end
    errorbar(ibias,mean(bvals(:,ibias),'omitnan'),std(bvals(:,ibias),'omitnan')/sqrt(nsubj_temp),'o','LineWidth',2);
end
text(1.5,-.2,sigstar(p),'FontSize',20);
XTick = 1:2;
set(gca,'xtick',XTick);
yline(0,':');
xlim([0 3]);
ylabel('Regression weight beta','FontSize',14);
legtxt = {'unbiased','biased'};
legend([legtxt],'Location','southeast');
title(sprintf('Effect of feedback value on reaction times\nError bars: SEM'),'FontSize',12);

%% 4a. Organize: RTs (z-scored) grouped on BIASED vs UNBIASED blocks

% Load epsilon-greedy percentages
load('out_fit_epsi.mat');
epsi(:,:,:,1) = [];
epsi(epsi<=.5) = 0;
epsi(epsi~=0) = 1;

rt_means_bias = nan(nsubj,2);
for isubj = 1:nsubj
    rts_vec1 = [];
    rts_vec2 = [];
    for icond = 1:nc
        for iq = 1:4
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            if epsi(isubj,icond,iq) == 0
                rts_vec1 = cat(1,rts_vec1,rts(blockrange,2:nt,icond,subjlist(isubj)));
            else
                rts_vec2 = cat(1,rts_vec2,rts(blockrange,2:nt,icond,subjlist(isubj)));
            end
        end 
    end
    
    % Get averages
    for ibias = 1:2
        if ibias == 1
            rts_vec = rts_vec1(:);
        else
            rts_vec = rts_vec2(:);
        end
        rt_means_bias(isubj,ibias) = mean(rts_vec);
    end
end
nonbiased_subjs = [];
for i = 1:4
    nonbiased_subjs = cat(2,nonbiased_subjs,epsi(:,:,i));
end
nonbiased_subjs = numel(find(sum(nonbiased_subjs,2) == 0));

[h,p] = ttest2(rt_means_bias(:,1),rt_means_bias(:,2),'VarType','unequal'); % Welch's t-test (for unequal sample size)

%% 4b. Plot: RTs (z-scored) grouped on BIASED vs UNBIASED blocks
figure;
hold on;
for ibias = 1:2
    if ibias == 2
        nsubj_temp = nsubj-nonbiased_subjs;
    else
        nsubj_temp = nsubj;
    end
    errorbar(ibias,mean(rt_means_bias(:,ibias),'omitnan'),std(rt_means_bias(:,ibias),'omitnan')/sqrt(nsubj_temp),'o','LineWidth',2);
end
text(1.5,.52,sigstar(p),'FontSize',20);
XTick = 1:2;
set(gca,'xtick',XTick);
xlim([0 3]);
ylabel('Reaction Time (s)','FontSize',14);
legtxt = {'unbiased','biased'};
legend([legtxt],'Location','southeast');
title(sprintf('Reaction times on (un)biased quarters\nError bars: SEM'),'FontSize',12);

%% 5a. Organize: Trial-wide RTs grouped on BIASED vs UNBIASED blocks

% This analysis looks at the average RT (z-scored) from the onset of the
% stimulus to the end of the trial across the 16 trials in a block for all subjects
% in all conditions, grouped by whether the subject was biased or unbiased in a given
% quarter. 

% Load epsilon-greedy percentages
load('out_fit_epsi.mat');
epsi(:,:,:,1) = [];
epsi(epsi<=.5) = 0;
epsi(epsi~=0) = 1;

rt_means_bias = nan(nsubj,nt,2);
for isubj = 1:nsubj
    
    rts_zscored = zscore(rts(:,:,:,subjlist(isubj)),0,'all'); % z-score RTs for the subject across entire experiment
    
    rts_vec1 = [];
    rts_vec2 = [];
    for icond = 1:nc
        for iq = 1:4
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            if epsi(isubj,icond,iq) == 0
                rts_vec1 = cat(1,rts_vec1,rts_zscored(blockrange,1:nt,icond));
            else
                rts_vec2 = cat(1,rts_vec2,rts_zscored(blockrange,1:nt,icond));
            end
        end 
    end
    
    % Get averages
    for ibias = 1:2
        if ibias == 1
            rts_vec = mean(rts_vec1,1);
        else
            rts_vec = mean(rts_vec2,1);
        end
        
        if isempty(rts_vec)
            rts_vec = nan(1,16);
        end
        rt_means_bias(isubj,:,ibias) = rts_vec;
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
    [h(it),p(it)] = ttest2(rt_means_bias(:,it,1),rt_means_bias(:,it,2),'VarType','unequal'); % Welch's t-test (for unequal sample size)
end

%% 5b. Plot: Trial-wide RTs grouped on BIASED vs UNBIASED blocks
figure;
hold on;
plotrgb = [.5 .5 .5;.8 .2 .2];
for ibias = 1:2
    if ibias == 2
        nsubj_temp = nsubj-nonbiased_subjs;
    else
        nsubj_temp = nsubj;
    end
    shadedErrorBar(1:nt,mean(rt_means_bias(:,:,ibias),1,'omitnan'),std(rt_means_bias(:,:,ibias),0,1,'omitnan')/sqrt(nsubj_temp),...
        'lineprops',{'Color',plotrgb(ibias,:),'LineWidth',2},'patchSaturation',0.075);
    for it = 1:nt
        scatter(it,mean(rt_means_bias(:,it,ibias),1,'omitnan'),60,'MarkerFaceColor',plotrgb(ibias,:),'MarkerFaceAlpha',.8,...
            'MarkerEdgeColor',[0 0 0],'MarkerEdgeAlpha',h(it),'LineWidth',1.5,'HandleVisibility','off');
    end
end
xticks([4 8 12 16]);
ylabel('reaction time (z)','FontSize',12);
legtxt = {'unbiased','biased'};
legend(legtxt,'Location','northeast');

%% 6a. Organize: RT on SWITCH trials accounting for general trial-wide RT trends

% This analysis looks at the average RT (z-scored) around switch trials, having
% subtracted general trends from the above analysis (5a). Switch trials that occur
% within the window of the main switch trial are disregarded.

% Note: Code section 5a must be run previous to this.

nt_bfr = 4; % number of trials around switch point (buffer trials)
rts_switch_means = nan(nsubj,nt_bfr*2+1);

for isubj = 1:nsubj
    % ignore excluded subjects
    resps_s     = resps(:,:,:,subjlist(isubj));
    rts_zscored = zscore(rts(:,:,:,subjlist(isubj)),0,'all');
    
    rts_switch = [];
    
    for ic = 1:nc
        for ib = 1:nb_c
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
                        range_l = 1;            % left RT range limit
                    else
                        bfr_l   = 0;
                        range_l = it-nt_bfr;
                    end
                    % NaNs for right side of switch
                    if nt-it-nt_bfr < 0
                        bfr_r   = -(nt-it-nt_bfr);  % right NaN buffer
                        range_r = nt;               % right RT range limit
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
                    % Subtract general effect of RTs based on trial position and bias
                    if biasflag
                        rt_means_normd = bsxfun(@minus,...
                            [nan(1,bfr_l) rts_zscored(ib,range_l:range_r,ic) nan(1,bfr_r)], ...
                            [nan(1,bfr_l) rt_means_bias(isubj,range_l:range_r,2) nan(1,bfr_r)]);
                    else
                        rt_means_normd = bsxfun(@minus,...
                            [nan(1,bfr_l) rts_zscored(ib,range_l:range_r,ic) nan(1,bfr_r)], ...
                            [nan(1,bfr_l) rt_means_bias(isubj,range_l:range_r,1) nan(1,bfr_r)]);
                    end
                    rt_means_normd(excl_tr) = NaN;
                    rts_switch = cat(1,rts_switch,rt_means_normd);
                end
            end
        end
    end
    rts_switch_means(isubj,:) = mean(rts_switch,1,'omitnan');
end

% Statistics
h = zeros(1,size(rts_switch_means,2));
p = h;
for it = 1:size(rts_switch_means,2)
    [h(it),p(it)] = ttest(rts_switch_means(:,it)); 
end

%% 6b. Plot: RTs around switches after accounting for general trends from bias/unbiased quarters
figure;
plotrgb = [.5 .5 .8];
hold on;
for ibias = 1:2
    shadedErrorBar(1:size(rts_switch_means,2),mean(rts_switch_means,1,'omitnan'),std(rts_switch_means,0,1,'omitnan')/sqrt(nsubj),...
        'lineprops',{'Color',plotrgb,'LineWidth',2},'patchSaturation',0.075);
    for it = 1:size(rts_switch_means,2)
        scatter(it,mean(rts_switch_means(:,it),1,'omitnan'),60,'MarkerFaceAlpha',.8,'MarkerFaceColor',plotrgb,...
            'MarkerEdgeColor',[0 0 0],'MarkerEdgeAlpha',h(it),'LineWidth',1.5,'HandleVisibility','off');
    end
end
xticklabels(-4:4);
ylabel('reaction time (z)','FontSize',12);
yline(0);
xline(5,'--');

%% local functions
function rgb = graded_rgb(ic,ib,nb)
    xc = linspace(.7,.2,nb);

    rgb =  [1,xc(ib),xc(ib); ...
               xc(ib),1,xc(ib); ...
               xc(ib),xc(ib),1];

    rgb = rgb(ic,:);
end
function stars = sigstar(p)
    if p <= .001
        stars = '***';
    elseif p <= .01
        stars = '**';
    elseif p <=.05
        stars = '*';
    else 
        stars = '';
    end
end
