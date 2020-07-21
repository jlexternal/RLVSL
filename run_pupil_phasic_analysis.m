% run_pupil_phasic_analysis.m
% 
% Usage: Standard analysis where pupil areas are compared with the variables of the
%        incident trial

clear all;
addpath('./Toolboxes/'); % add path containing ANOVA function
%%

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
    cfg.incl_nan = false;
end

for icond = 1:nc
    if usecfg
        epc_struct{icond} = pupil_get_epochs(subjlist,condtypes{icond},cfg);
    else
        epc_struct{icond} = pupil_get_epochs(subjlist,condtypes{icond});
    end
end

% choose the smallest epoch window for comparison
epc_range    = min([epc_struct{1}.epoch_window; epc_struct{2}.epoch_window; epc_struct{3}.epoch_window],[],1);
epc_fb_onset = [epc_struct{1}.epoch_window(1); epc_struct{2}.epoch_window(1); epc_struct{3}.epoch_window(1)]+1;

ifig = 0;

%% 1. Organize: Regression beta of feedback value as predictor of pupil area along the epoch
%        Comparison between conditions

iszscored = true; % set to true to z-score the pupil areas for any subject
pcchanged = false;  % set pupil values to percent change
baselined = true;  % baseline each epoch to before the fb onset
area2diam = false;  % convert areas to diameters (to see linear change)
pumeasure = 'area';
zs_str = '';
for icond = 1:nc
    epochs       = epc_struct{icond}.epochs;
    plot_window  = epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2);
    epoch_window = epc_struct{icond}.epoch_window;
    epochs       = epochs(:,plot_window);
    idx_subj     = epc_struct{icond}.idx_subj;
    fbs          = epc_struct{icond}.fbs/100;
    if icond == 1
        bvals = zeros(nsubj,size(epochs,2),nc);
        err_bar = zeros(icond,size(bvals,2));
    end
    
    % Subject-level mean analysis
    for isubj = 1:nsubj
        ind_subj = idx_subj == isubj; % identify which epochs to analyze for the subject
        if area2diam
            epochs(ind_subj,:) = epochs(ind_subj,:) + abs(min(epochs(ind_subj,:),[],'all')); 
            epochs(ind_subj,:) = sqrt(epochs(ind_subj,:)); 
            pumeasure = 'diameter';
        end
        if iszscored
            epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
            zs_str = '(z-scored pupil area)';
        end
        if baselined
            epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),mean(epochs(ind_subj,1:epoch_window(1)),2));
        end
        if pcchanged
            %basevalue = mean(epochs(ind_subj,1:epoch_window(1)),2); % percent change from baselined value
            basevalue = epochs(ind_subj,epoch_window(1)+1);
            epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),basevalue)./basevalue; % percent change
            zs_str = '(percent change in pupil area from onset)';
        end
        
        n_eps = size(epochs(ind_subj,:),1);
        for isamp = 1:size(epochs,2)
            
            bval = regress(epochs(ind_subj,isamp),[ones(n_eps,1) fbs(ind_subj)]);
            bvals(isubj,isamp,icond) = bval(2);
        end
    end
    err_bar(icond,:) = std(bvals(:,:,icond),[],1)/sqrt(nsubj);
    
end

mrk_pupil_betas = nan(1,size(bvals,2));

%%      1. Statistics: Regression beta between conditions
% 
% Design:
%        N subjects, 2 within-subject factors:  1/ 4 levels (quarters)
%                                               2/ 3 levels (conditions)
%        No between-subject factors

tbl_pupil_betas	= cell(1,size(bvals,2));
sig_pupil_betas = zeros(1,size(bvals,2));
for ie = 1:size(bvals,2)
    tbl_pupil_betas{ie} = simple_mixed_anova(reshape(bvals(:,ie,:),[nsubj,3]),[],{'Condition'});
    col_sigp_betas = tbl_pupil_betas{ie}.(5);
    sig_pupil_betas(ie) = col_sigp_betas(3); % quarter, condition, quarter x condition
    mrk_pupil_betas(ie) = sig_pupil_betas(ie) < .05;
end
mrk_pupil_betas(mrk_pupil_betas==0) = nan;

%%      1. Plot: Regression beta of feedback value as predictor of pupil area along the epoch
xaxis = ((1:size(bvals,2))-epoch_window(1))*2/1000;
ifig = ifig+1;
figure(ifig);
for icond = 1:nc
    shadedErrorBar(xaxis,mean(bvals(:,:,icond),1),err_bar(icond,:),'lineprops',{'LineWidth',2,'Color',graded_rgb(icond,1,1)},'patchSaturation',0.075);
    hold on;
end
sig_betas = scatter(xaxis,mrk_pupil_betas*.5,'s','filled');
shade_str = 'SEM';
legend([condtypes,'sig:Condition (ANOVA)'],'Location','southwest');
xline(0,'LineWidth',2,'HandleVisibility','off');
xlabel('Time around fb onset (s)');
yline(0,'HandleVisibility','off');
yline(.5,':','HandleVisibility','off');
yline(-.5,':','HandleVisibility','off');
yline(1,':','HandleVisibility','off');
yline(-1,':','HandleVisibility','off');
ylabel('Regression Beta');
xlim([min(xaxis) max(xaxis)]);
ylim([-2.5 1.1]);
baseline_txt = '(not baselined)';
if baselined
    baseline_txt = '(baselined)';
end
title_str = sprintf('Regr. coef. beta of feedback as a predictor of %s pupil %s %s \naround fb onset across all subjs & conditions \nShaded area: %s',...
                    baseline_txt,pumeasure,zs_str,shade_str);
title(title_str);

%% 2. Organize: Pupil dilation across quarters

iszscored = true;
baselined = true;

for icond = 1:nc
    epochs       = epc_struct{icond}.epochs;
    plot_window  = (epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2));
    epoch_window = epc_struct{icond}.epoch_window;
    epochs       = epochs(:,plot_window);
    idx_subj     = epc_struct{icond}.idx_subj;
    qts          = epc_struct{icond}.qts;
    fbs          = epc_struct{icond}.fbs/100;
    
    if icond == 1
        epoch_means = zeros(nsubj,size(epochs,2),4,nc); % (subjects,epochlength,quarters,conditions)
        bvals_qt  	= zeros(nsubj,size(epochs,2),4,nc); 
    end
    
    for isubj = 1:nsubj
        ind_subj = idx_subj == isubj;
        if iszscored
            epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
            zs_str = '(z-scored pupil area)';
        end
        if baselined
            epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),mean(epochs(ind_subj,1:epoch_window(1)),2));
        end
        
        for iq = 1:4
            
            epoch_means(isubj,:,iq,icond) = mean(epochs(ind_subj & qts==iq,:),1);

            n_eps = size(epochs(ind_subj & qts==iq,:),1);
            for isamp = 1:size(epochs,2)
                bval = regress(epochs(ind_subj & qts==iq,isamp),[ones(n_eps,1) fbs(ind_subj & qts==iq)]);
                bvals_qt(isubj,isamp,iq,icond) = bval(2);
                
                % run anova with factor : quarter
                
                % Epiphany analysis: 
                % find epiphany points in structured condition at which quarter, use random
                % as control but with the same bins (before and after epiphany)
                
                % cognitive cost in RL systems 
            end
        end
    end
end

%%      2. Statistics: Pupil dilation across quarters across conditions
%
% Design:
%        N subjects, 2 within-subject factors:  1/ 4 levels (quarters)
%                                               2/ 3 levels (conditions)
%        No between-subject factors

% Note: - Assuming sphericity between factors

tbl_pupil       = cell(1,size(epochs,2));
tbl_pupil_cond  = cell(nc,size(epochs,2));
tbl_betas       = cell(1,size(epochs,2));
tbl_betas_qt    = cell(3,size(epochs,2));
sig_pupil       = zeros(3,size(epochs,2));
sig_pupil_cond  = zeros(1,size(epochs,2),nc);
sig_betas       = zeros(3,size(epochs,2));
sig_betas_qt    = zeros(1,size(epochs,2),nc);
mrk_pupil       = nan(3,size(epochs,2));
mrk_pupil_cond  = nan(1,size(epochs,2),nc);
mrk_betas       = nan(3,size(epochs,2));
mrk_betas_qt    = nan(1,size(epochs,2),nc);
for ie = 1:size(epochs,2)
%    tbl_pupil{ie} = simple_mixed_anova(reshape(epoch_means(:,ie,:,:),[nsubj,4,3]),[],{'Quarter','Condition'});
%    tbl_betas{ie} = simple_mixed_anova(reshape(bvals_qt(:,ie,:,:),[nsubj,4,3]),[],{'Quarter','Condition'});    
    for icond = 1:nc
%         tbl_pupil_cond{icond,ie} = simple_mixed_anova(reshape(epoch_means(:,ie,:,icond),[nsubj,4]),[],{'Quarter'});
%         col_pup_sigp_cond = tbl_pupil_cond{icond,ie}.(5);
%         sig_pupil_cond(:,ie,icond) = col_pup_sigp_cond(3); % quarter
%         mrk_pupil_cond(:,ie,icond) = sig_pupil_cond(:,ie,icond) < .05;
        
        tbl_betas_qt{icond,ie} = simple_mixed_anova(reshape(bvals_qt(:,ie,:,icond),[nsubj,4]),[],{'Quarter'});
        col_bet_sigp_cond = tbl_betas_qt{icond,ie}.(5);
        sig_betas_qt(:,ie,icond) = col_bet_sigp_cond(3);
        mrk_betas_qt(:,ie,icond) = sig_betas_qt(:,ie,icond) < .05;
    end
    
%     col_sigp = tbl_pupil{ie}.(5);
%     sig_pupil(:,ie) = col_sigp([3,5,7]); % quarter, condition, quarter x condition
%     mrk_pupil(:,ie) = sig_pupil(:,ie) < .05;
%     
%     col_sigb = tbl_betas{ie}.(5);
%     sig_betas(:,ie) = col_sigb([3,5,7]);
%     mrk_betas(:,ie) = sig_betas(:,ie) < .05;
end

mrk_pupil(mrk_pupil==0) = nan;
mrk_pupil_cond(mrk_pupil_cond==0) = nan;
mrk_betas(mrk_betas==0) = nan;
mrk_betas_qt(mrk_betas_qt==0) = nan;

%%      2. Plot: Pupil dilation (z-scored) across quarters 
%  Grouped by condition

xaxis = ((1:size(epochs,2))-epoch_window(1))*2/1000;
ifig = ifig+1;
for icond = 1:nc   
    for ip = 1:2
        vishandle = 'on';
        if ip == 2 % subplots by condition
            figure(ifig);
            hold on;
            subplot(3,1,icond);
            lwidth = 2;
            quar_sig_cond = scatter(xaxis,mrk_pupil_cond(1,:,icond)*.55,'s','filled','HandleVisibility',vishandle);
        else % all conditions
            if ismember(icond,[2 3])
                vishandle = 'off';
            end
            figure(ifig+1);
            hold on;
            lwidth = 1;
            
            quar_sig    = scatter(xaxis,mrk_pupil(1,:)*.55,'s','filled','HandleVisibility',vishandle);
            cond_sig    = scatter(xaxis,mrk_pupil(2,:)*.5,'s','filled','HandleVisibility',vishandle);
            ntrxn_sig   = scatter(xaxis,mrk_pupil(3,:)*.45,'s','filled','HandleVisibility',vishandle);
        end
        xline(0,'LineWidth',2,'HandleVisibility','off');
        yline(0,'HandleVisibility','off');
        for yline_val = [.4 .2 -.2 -.4]
            yline(yline_val,':','HandleVisibility','off');
        end

        for iq = 1:4
            shadedErrorBar(xaxis,mean(epoch_means(:,:,iq,icond),1),std(epoch_means(:,:,iq,icond))/sqrt(nsubj),...
                        'lineprops',{'Color',graded_rgb(icond,iq,4),'LineWidth',lwidth},'patchSaturation',0.075);
        end

        leg_txt = ["Q1","Q2","Q3","Q4"];
        if ip == 2
            legend(["sig:Quarter (ANOVA)",leg_txt],'Location','eastoutside');
            title(sprintf(['Evolution of pupil area around onset of feedback split across quarters\n(%s condition); %d subjects,\n' ...
                        'shaded error: SEM'],condtypes{icond},nsubj));
        else
            sigbar_txt = ["sig:Quarter (ANOVA)","sig:Condition (ANOVA)","sig:Quarter x Condition (ANOVA)"];
            legend([sigbar_txt repmat(leg_txt,[1 3])],'Location','eastoutside');
            legend('boxoff');
            title(sprintf(['Evolution of pupil area around onset of feedback split across quarters\n(all conditions); %d subjects,\n' ...
                        'shaded error: SEM'],nsubj));
        end
        xlabel('Time around fb onset (s)');
        ylabel('Z-scored pupil area (a.u.)');
        xlim([min(xaxis) max(xaxis)]);
        
    end
end

%%      2. Plot: Regression beta of feedback on pupil dilation (z-scored) across quarters 
%  Grouped by condition

xaxis = ((1:size(epochs,2))-epoch_window(1))*2/1000;
ifig = ifig+1;
vishandle = 'on';
for icond = 1:nc
    for ip = 2
        if ip == 2
            figure(ifig);
            hold on;
            subplot(3,1,icond);
            lwidth = 2;
        else
            figure(ifig+1);
            hold on;
            lwidth = 1;
            
            if icond == 2
                vishandle = 'off';
            end
            if false
            quar_sig    = scatter(xaxis,mrk_betas(1,:)*2,'s','filled','HandleVisibility',vishandle);
            cond_sig    = scatter(xaxis,mrk_betas(2,:)*1.5,'s','filled','HandleVisibility',vishandle);
            ntrxn_sig   = scatter(xaxis,mrk_betas(3,:)*1,'s','filled','HandleVisibility',vishandle);
            end
        end
        
        xline(0,'LineWidth',2,'HandleVisibility','off');
        yline(0,'HandleVisibility','off');
        for yline_val = [-1 -.5 .5 1]
            yline(yline_val,':','HandleVisibility','off');
        end

        for iq = 1:4
            shadedErrorBar(xaxis,mean(bvals_qt(:,:,iq,icond),1),std(bvals_qt(:,:,iq,icond))/sqrt(nsubj),...
                            'lineprops',{'Color',graded_rgb(icond,iq,4),'LineWidth',2},'patchSaturation',0.075);
        end
        hold on;
        if false
        quar_sig_qt    = scatter(xaxis,mrk_betas_qt(1,:,icond)*2,'s','filled','HandleVisibility',vishandle);
        end
        
        leg_txt = ["Q1","Q2","Q3","Q4"];
        if ip == 2
            legend(leg_txt,'Location','southeast');
            title(sprintf('Condition: %s',condtypes{icond}));
        else
            sigbar_txt = ["sig:Quarter (ANOVA)","sig:Condition (ANOVA)","sig:Quarter x Condition (ANOVA)"];
            legend([sigbar_txt repmat(leg_txt,[1 3])],'Location','eastoutside');
            legend('boxoff');
            title(sprintf(['Evolution of reg. beta of fb around onset of feedback split across quarters\n(all conditions); %d subjects,\n' ...
                        'shaded error: SEM'],nsubj));
        end
        xlabel('Time around fb onset (s)');
        ylabel('Regression beta');
        xlim([min(xaxis) max(xaxis)]);
        ylim([-2.5 1.1]);
    end 
end
sgtitle(sprintf(['Evolution of reg. beta of fb around onset of feedback split across quarters\n %d subjects; ' ...
                            'shaded error: SEM'],nsubj));
          
%% 3. Organize: Regression beta of feedback value as predictor of pupil area in the 1st and 2nd half of blocks between 1st and 2nd half of experiment
%        Comparison within conditions 

iszscored = true;  % set to true to z-score the pupil areas for any subject
baselined = true;

zs_str = '';
for icond = 1:nc
    epochs       = epc_struct{icond}.epochs;
    plot_window  = epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2);
    epoch_window = epc_struct{icond}.epoch_window;
    epochs       = epochs(:,plot_window);
    idx_subj     = epc_struct{icond}.idx_subj;
    fbs          = epc_struct{icond}.fbs/100;
    trs          = epc_struct{icond}.trs;
    qts          = epc_struct{icond}.qts;
    
    if icond == 1
        bvals = zeros(nsubj,size(epochs,2),nc,2,2); % data structure dims(subject,timesample,condition,half_exp,half_block)
        err_bar = zeros(nc,size(bvals,2),2,2);
    end
    
    % Subject-level mean analysis
    for iq = [1 2] % split experiment into first and last half
        for isubj = 1:nsubj
            ind_subj = idx_subj == isubj; % identify which epochs to analyze for the subject
            ind_half1 = trs <= 8;         % epochs of 1st half of block
            ind_half2 = trs > 8;          % epochs of 2nd half of block
            ind_qt = qts == 2*(iq-1)+1 | qts == 2*(iq-1)+2; % which epochs belong to which half of experiment

            if baselined
                epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),mean(epochs(ind_subj,1:epoch_window(1)),2));
            end
        
            if iszscored
                epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
                zs_str = '(z-scored pupil area)';
            end

            
            ind_ep1 = ind_subj & ind_half1 & ind_qt;
            ind_ep2 = ind_subj & ind_half2 & ind_qt;
            n_eps1 = size(epochs(ind_ep1,:),1);
            n_eps2 = size(epochs(ind_ep2,:),1);
            for isamp = 1:size(epochs,2)
                bval1 = regress(epochs(ind_ep1,isamp),[ones(n_eps1,1) fbs(ind_ep1)]);
                bval2 = regress(epochs(ind_ep2,isamp),[ones(n_eps2,1) fbs(ind_ep2)]);
                bvals(isubj,isamp,icond,iq,1) = bval1(2); % 1st half betas
                bvals(isubj,isamp,icond,iq,2) = bval2(2); % 2nd half betas
            end
        end
        err_bar(icond,:,iq,1) = std(bvals(:,:,icond,iq,1),[],1)/sqrt(nsubj);
        err_bar(icond,:,iq,2) = std(bvals(:,:,icond,iq,2),[],1)/sqrt(nsubj);
    end
    
end

%%      3. Plot: Regression beta of feedback value as predictor of pupil area in the 1st and 2nd half of blocks between 1st and 2nd half of experiment

xaxis = ((1:size(bvals,2))-epoch_window(1))*2/1000;

ifig = ifig+1;
figure(ifig);
hold on;
for icond = 1:nc
    for iq = 1:2
        subplot(1,2,iq);
        shadedErrorBar(xaxis,mean(bvals(:,:,icond,iq,1),1),err_bar(icond,:,iq,1),...
                        'lineprops',{'LineWidth',2,'Color',graded_rgb(icond,1,1)},'patchSaturation',0.075);
        shadedErrorBar(xaxis,mean(bvals(:,:,icond,iq,2),1),err_bar(icond,:,iq,2),...
                        'lineprops',{':','LineWidth',2,'Color',graded_rgb(icond,1,1)},'patchSaturation',0.075);
        xline(0,'LineWidth',2,'HandleVisibility','off');
        xlabel('Time around fb onset (s)');
        yline(0,'HandleVisibility','off');
        ylabel('Regression Beta');
        ylim([-3 3]);
        xlim([min(xaxis) max(xaxis)]);
        title(sprintf('Experiment half: %d',iq));
    end
end
leg_str = [strcat(condtypes,' 1st half'); strcat(condtypes,' 2nd half')];
leg_str = leg_str(:);
shade_str = 'SEM';
legend(leg_str);
title_str = sprintf('Regr. coef. beta of feedback as a predictor of %s pupil area \naround fb onset across all subjs & conditions\nShaded bars: %s',...
                    zs_str,shade_str);
sgtitle(title_str);

%% 4. Organize: Pupil dilation at feedback upon making consistent/inconsistent choice after previous trial
%   i.e. consistency of outcome w.r.t. correct choice
iszscored = true; % set to true to z-score the pupil areas for any subject
baselined = false;  % baseline each epoch to before the fb onset

epochs_con_means = zeros(nsubj,sum(epc_range)+1,nc);
epochs_inc_means = zeros(nsubj,sum(epc_range)+1,nc);
for icond = 1:nc
    epochs       = epc_struct{icond}.epochs;
    plot_window  = epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2);
    epoch_window = epc_struct{icond}.epoch_window;
    epochs       = epochs(:,plot_window);
    idx_subj     = epc_struct{icond}.idx_subj;
    fbs          = epc_struct{icond}.fbs/100;
    trs          = epc_struct{icond}.trs;
    rsp          = epc_struct{icond}.rsp;
    
    epochs_con = [];
    epochs_inc = [];
    
    fprintf('Organizing over condition %d',icond);
    istart = 1;
    nt = max(trs);
    for isubj = 1:nsubj
        ind_subj = idx_subj == isubj; % identify which epochs to analyze for the subject
        
        if true
            epochs(ind_subj,:) = epochs(ind_subj,:) + abs(min(epochs(ind_subj,:),[],'all')); 
            epochs(ind_subj,:) = sqrt(epochs(ind_subj,:)); 
        end
        
        if baselined
            epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),mean(epochs(ind_subj,1:epoch_window(1)),2));
        end

        if iszscored
            epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
            zs_str = '(z-scored pupil area)';
        end
        
        iend = istart+length(idx_subj(idx_subj==isubj))-1;
        for i = istart:iend
            if i == istart || trs(i) == 1 % ignore first trial(s)
                continue
            else
                if trs(i)-trs(i-1) == 1 % consecutive trials maintained
                    if (fbs(i-1) > 0 && rsp(i) == 1) || (fbs(i-1) < 0 && rsp(i) == 2) % consistent
                        epochs_con = cat(1,epochs_con,epochs(i,:));
                    else % inconsistent
                        epochs_inc = cat(1,epochs_inc,epochs(i,:));
                    end
                end
            end
        end
        istart = istart+length(idx_subj(idx_subj==isubj));
        epochs_con_means(isubj,:,icond) = mean(epochs_con,1);
        epochs_inc_means(isubj,:,icond) = mean(epochs_inc,1);
    end
end

%%      4. Plot: Pupil dilation at feedback upon making consistent/inconsistent choice after previous trial
xaxis = ((1:size(epochs,2))-epoch_window(1))*2/1000;
figure;
hold on;
for icond = 1:3
    subplot(3,1,icond);
    shadedErrorBar(xaxis,mean(epochs_con_means(:,:,icond),1),std(epochs_con_means(:,:,icond))/sqrt(nsubj),...
                        'lineprops',{'Color',graded_rgb(icond,3,3),'LineWidth',2},'patchSaturation',0.075);
    hold on;
    shadedErrorBar(xaxis,mean(epochs_inc_means(:,:,icond),1),std(epochs_inc_means(:,:,icond))/sqrt(nsubj),...
                        'lineprops',{':','Color',graded_rgb(icond,3,3),'LineWidth',2},'patchSaturation',0.075);
    xline(0,'LineWidth',2,'HandleVisibility','off');
    yline(0,'HandleVisibility','off');
    xlabel('Time around fb onset (s)');
    ylabel('Z-scored pupil area (a.u.)');
    xlim([min(xaxis) max(xaxis)]);
end

%% 5. Organize: Regression beta of pupil area and feedback magnitude around EPIPHANY within conditions

run_epiphany_qts; % get epiphany quarters for relevant subjects

% Pupil analysis options
baselined = true;   % baseline each epoch to before the fb onset
iszscored = true;   % set to true to z-score the pupil areas for any subject
zs_str = '';
nexcl = [0 0];

for icond = 1:2
    % compare each structured condition to its random counterpart
    epochs       = epc_struct{icond}.epochs;
    plot_window  = epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2);
    epoch_window = epc_struct{icond}.epoch_window;
    epochs       = epochs(:,plot_window);
    idx_subj     = epc_struct{icond}.idx_subj;
    fbs          = epc_struct{icond}.fbs/100;
    qts          = epc_struct{icond}.qts;
    % load rnd condition
    epochs_rnd   = epc_struct{3}.epochs;
    idx_subj_rnd = epc_struct{3}.idx_subj;
    fbs_rnd      = epc_struct{3}.fbs/100;
    qts_rnd      = epc_struct{3}.qts;
    
    if icond == 1
        bvals   = zeros(nsubj,size(epochs,2),2,2,2); % subjs, epoch, rep/alt, cond/rnd, epiph
    end
    
    isubj = 0;
    excl_epiph = [];
    for subj = subjlist
        isubj = isubj + 1;
        ind_subj     = idx_subj == isubj;
        ind_subj_rnd = idx_subj_rnd == isubj;
        
        if iszscored
            epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
            epochs_rnd(ind_subj_rnd,:) = zscore(epochs_rnd(ind_subj_rnd,:),[],'all'); % z-score all epochs from a subject
            zs_str = '(z-scored pupil area)';
        end
        if baselined
            epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),mean(epochs(ind_subj,1:epoch_window(1)),2));
            epochs_rnd(ind_subj_rnd,:) = bsxfun(@minus,epochs_rnd(ind_subj_rnd,:),mean(epochs_rnd(ind_subj_rnd,1:epoch_window(1)),2));
        end
        
        % pre and post epiphany analysis
        if icond == 1
            if ~isempty(prepost_epiph_rep{subj,1})
                pre_ep_qts = prepost_epiph_rep{subj,1}; 
            else
                excl_epiph = [excl_epiph isubj]; % exclude those who did not have epiphany in repeating cond.
                continue
            end
        elseif icond == 2
            if ~isempty(prepost_epiph_alt{subj,1})
                pre_ep_qts = prepost_epiph_alt{subj,1}; 
            else 
                excl_epiph = [excl_epiph isubj]; % exclude those who did not have epiphany in alternating cond.
                continue
            end
        end
        
        idx_qt       = pre_ep_qts; % select quarters
        
        % Regression analysis before and after epiphany
        for iepiph = 1:2
            if iepiph == 2 % after epiphany
                idx_qt = setdiff(1:4,idx_qt);
            end
            
            n_eps = size(epochs(ind_subj & ismember(qts,idx_qt),:),1);
            n_eps_rnd = size(epochs_rnd(ind_subj_rnd & ismember(qts_rnd,idx_qt),:),1);
            for isamp = 1:size(epochs,2)
                % structured condition
                bval = regress(epochs(ind_subj & ismember(qts,idx_qt),isamp),[ones(n_eps,1) fbs(ind_subj & ismember(qts,idx_qt))]);
                bvals(isubj,isamp,icond,1,iepiph) = bval(2);
                
                % random condition
                bval = regress(epochs_rnd(ind_subj_rnd & ismember(qts_rnd,idx_qt),isamp),[ones(n_eps_rnd,1) fbs_rnd(ind_subj_rnd & ismember(qts_rnd,idx_qt))]);
                bvals(isubj,isamp,icond,2,iepiph) = bval(2);
            end
        end
    end
    nexcl(icond) = numel(excl_epiph);
end

bvals(bvals == 0) = NaN;

%%      5. Plot: Regression beta of pupil area and feedback magnitude around EPIPHANY within conditions

xaxis = ((1:size(epochs,2))-epoch_window(1))*2/1000;
ips = [1 2; 3 4];
ifig = ifig+1;
figure;
hold on;
sgtitle('Evolution of reg. beta of fb around onset of feedback split around epiphany');
for icond = 1:2
    for iepiph = 1:2
        if iepiph == 2  % post-epiphany
            linetype = '-';
        else            % pre-epiphany
            linetype = ':';
        end
        subplot(4,1,ips(icond,1));
        shadedErrorBar(xaxis,mean(bvals(:,:,icond,1,iepiph),1,'omitnan'),std(bvals(:,:,icond,1,iepiph),'omitnan')/sqrt(nsubj-nexcl(icond)),...
                            'lineprops',{linetype,'Color',graded_rgb(icond,3,3),'LineWidth',2},'patchSaturation',0.075);
        hold on;
        
        subplot(4,1,ips(icond,2));
        shadedErrorBar(xaxis,mean(bvals(:,:,icond,2,iepiph),1,'omitnan'),std(bvals(:,:,icond,2,iepiph),'omitnan')/sqrt(nsubj-nexcl(icond)),...
                            'lineprops',{linetype,'Color',graded_rgb(3,3,3),'LineWidth',2},'patchSaturation',0.075);
        hold on;
    end
end

for iplot = 1:4
    subplot(4,1,iplot);
    xline(0,'LineWidth',2,'HandleVisibility','off');
    yline(0,'HandleVisibility','off');
    yline(.5,':','HandleVisibility','off');
    yline(-.5,':','HandleVisibility','off');
    yline(1,':','HandleVisibility','off');
    yline(-1,':','HandleVisibility','off');
    
    xlabel('Time around fb onset (s)');
    ylabel('Regression beta');
    xlim([min(xaxis) max(xaxis)]);
    ylim([-2.5 1.1]);
    legend({'Pre-epiphany','Post-epiphany'},'Location','southeast');
    
    if iplot == 1
        title(sprintf('Condition: %s\n %d/%d subjects',condtypes(1),nsubj-nexcl(1),nsubj));
    elseif iplot == 3
        title(sprintf('Condition: %s\n %d/%d subjects',condtypes(2),nsubj-nexcl(2),nsubj));
    elseif iplot == 2
        title(sprintf('Condition: rnd (compare to %s)',condtypes(1)));
    else
        title(sprintf('Condition: rnd (compare to %s)',condtypes(2)));
    end
end

%% 6. Organize: Regression beta of feedback on pupil area on BIASED and UNBIASED blocks across ALL CONDITIONS

% Load epsilon-greedy percentages
load('out_fit_epsi.mat');
epsi(:,:,:,1) = [];
epsi(epsi<=.5) = 0;
epsi(epsi~=0) = 1;

iszscored = true; % set to true to z-score the pupil areas for any subject
baselined = false;  % baseline each epoch to before the fb onset
zs_str = '';

% Group all pupil epochs together
epochs       = [];
idx_subj     = [];
fbs          = [];
qts          = [];
cds          = []; % conditions
for icond = 1:nc
    epochs_cond  = epc_struct{icond}.epochs;
    plot_window  = epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2);
    epoch_window = epc_struct{icond}.epoch_window;
    epochs       = cat(1,epochs,epochs_cond(:,plot_window));
    idx_subj     = cat(1,idx_subj,epc_struct{icond}.idx_subj);
    fbs          = cat(1,fbs,epc_struct{icond}.fbs/100);
    qts          = cat(1,qts,epc_struct{icond}.qts);
    cds          = cat(1,cds,icond.*ones(size(epc_struct{icond}.qts)));
end
idx_bias = zeros(size(qts));

bvals = zeros(nsubj,size(epochs,2),2); %(isubj,epoch,ibias)
err_bar = zeros(icond,size(bvals,2),2);
epoch_means = nan(nsubj,sum(epc_range)+1,2);

epochs_test = epochs;
% Subject-level mean analysis
for isubj = 1:nsubj
    ind_subj = idx_subj == isubj; % identify which epochs to analyze for the subject
    if iszscored
        epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
        zs_str = '(z-scored pupil area)';
    end
    if baselined
        epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),mean(epochs(ind_subj,1:epoch_window(1)),2));
    end

    % Identify quarters that are biased/unbiased
    for icond = 1:3
        for iq = 1:4
            if epsi(isubj,icond,iq) == 1
                idx_bias(cds==icond & qts==iq & idx_subj == isubj) = 1; % 1 if biased, 0 if unbiased
            end
        end
    end

    % Regress
    for biased = 1:2 % 1:unbiased, 2:biased
        qts_biased = biased-1;
        n_eps = size(epochs(ind_subj & idx_bias==qts_biased,:),1);
        disp(sum(fbs(ind_subj & idx_bias==qts_biased)))
        for isamp = 1:size(epochs,2)
            bval = regress(epochs(ind_subj & idx_bias==qts_biased,isamp),[ones(n_eps,1) fbs(ind_subj & idx_bias==qts_biased)]);
            % Note: Rank deficient warnings come from subjects who do not make biased decisions
            bvals(isubj,isamp,biased) = bval(2);
        end
        
        epoch_means(isubj,:,biased) = mean(epochs(ind_subj & idx_bias==qts_biased,:),1);
        
        %{
        %testing
        xaxis = ((1:size(bvals,2))-epoch_window(1))*2/1000;
        clf;
        for i = 1:2
            shadedErrorBar(xaxis,mean(bvals(isubj,:,i),1,'omitnan'),std(bvals(isubj,:,i),0,1,'omitnan'),...
                'lineprops',{'LineWidth',2},'patchSaturation',0.075);
            hold on;
        end
        pause;
        %}
    end
end
% Account for subjects that are never biased
bvals(bvals == 0)=nan;
nonbiased_subjs = [];
for i = 1:4
    nonbiased_subjs = cat(2,nonbiased_subjs,epsi(:,:,i));
end
nonbiased_subjs = numel(find(sum(nonbiased_subjs,2) == 0));
for ibias = 1:2
    if ibias == 2
        nsubj_temp = nsubj-nonbiased_subjs;
    else
        nsubj_temp = nsubj;
    end
    err_bar(icond,:,ibias) = std(bvals(:,:,ibias),[],1,'omitnan')/sqrt(nsubj_temp);
end

%%      6. Plot: Regression beta of feedback on pupil area on BIASED and UNBIASED blocks across ALL CONDITIONS
xaxis = ((1:size(bvals,2))-epoch_window(1))*2/1000;
ifig = ifig+1;
figure(ifig);
for ibias = 1:2
    figure(1)
    shadedErrorBar(xaxis,mean(bvals(:,:,ibias),1,'omitnan'),err_bar(icond,:,ibias),'lineprops',{'LineWidth',2},'patchSaturation',0.075);
    hold on;
    shade_str = 'SEM';
    legtxt = {'unbiased','biased'};
    ylabel('Regression Beta');
    figure(2);
    shadedErrorBar(xaxis,mean(epoch_means(:,:,ibias),1,'omitnan'),std(epoch_means(:,:,ibias),0,1,'omitnan')/sqrt(nsubj_temp),'lineprops',{'LineWidth',2},'patchSaturation',0.075);
    shade_str = 'SEM';
    legtxt = {'unbiased','biased'};
    ylabel('Z-score');
end

for i =1:2
    figure(i)
    legend([legtxt],'Location','southwest');
    xline(0,'LineWidth',2,'HandleVisibility','off');
    xlabel('Time around fb onset (s)');
    yline(0,'HandleVisibility','off');
    xlim([min(xaxis) max(xaxis)]);
end


%% Plot: Pupil dilation grouped in to pos and neg feedback

iszscored = true;
baselined = true;

ifig = ifig+1;
figure(ifig);
for icond = 1:nc
    epochs = epc_struct{icond}.epochs;
    plot_window = (epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2));
    epoch_window = epc_struct{icond}.epoch_window;
    epochs = epochs(:,plot_window);
    idx_subj = epc_struct{icond}.idx_subj;
    fbs = epc_struct{icond}.fbs/100;
    
    epoch_mean_pos = [];
    epoch_mean_neg = [];

    for isubj = 1:nsubj
        ind_subj = idx_subj == isubj;
        
        if baselined
            epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),mean(epochs(ind_subj,1:epoch_window(1)),2));
        end

        if iszscored
            epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
            zs_str = '(z-scored pupil area)';
        end
        
        epoch_mean_pos(isubj,:) = mean(epochs(ind_subj & fbs>=0,:),1);
        epoch_mean_neg(isubj,:) = mean(epochs(ind_subj & fbs<0,:),1);
    end

    xaxis = ((1:size(epochs,2))-epoch_window(1))*2/1000;
    subplot(3,1,icond);
    yline(.4,':','HandleVisibility','off');
    yline(-.4,':','HandleVisibility','off');
    shadedErrorBar(xaxis,mean(epoch_mean_pos,1),std(epoch_mean_pos)/sqrt(nsubj),...
                    'lineprops',{'Color',[.8 .2 .2],'LineWidth',2},'patchSaturation',0.075);
    hold on;
    shadedErrorBar(xaxis,mean(epoch_mean_neg,1),std(epoch_mean_neg)/sqrt(nsubj),...
                    'lineprops',{'Color',[.2 .2 .8],'LineWidth',2},'patchSaturation',0.075);
    legend('pos. fb','neg. fb');
    xline(0,'LineWidth',2,'HandleVisibility','off');
    yline(0,'HandleVisibility','off');
    xlabel('Time around fb onset (s)');
    ylabel('Z-scored pupil area (a.u.)');
    xlim([min(xaxis) max(xaxis)]);
    ylim([-.6 .6]);
    title(sprintf(['Evolution of pupil area around onset of feedback\n(%s condition); %d subjects,\n' ...
                    'shaded error: SEM'],condtypes{icond},nsubj));
end

%% Local functions
function rgb = graded_rgb(ic,iq,nq)
    xc = linspace(.8,.2,nq);

    rgb =  [1,xc(iq),xc(iq); ...
               xc(iq),1,xc(iq); ...
               xc(iq),xc(iq),1];

    rgb = rgb(ic,:);
end
