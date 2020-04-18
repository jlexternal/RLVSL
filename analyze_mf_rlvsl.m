% Model-free analysis of behavior on experiment RLVSL
%
clear all;
close all;

%% Load data
ifig = 1;
nsubjtot    = 31;
excluded    = [1];
subjlist    = setdiff(1:nsubjtot, excluded);
subparsubjs = [excluded 11 23 28];
subjlist = setdiff(1:nsubjtot, subparsubjs); % if excluding underperforming/people who didn't get it
% load experiment structure
nsubj = numel(subjlist);
% Data manip step
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',2,2);
load(filename,'expe');
nt = expe(1).cfg.ntrls;
nb = expe(1).cfg.nbout;
ntrain = expe(1).cfg.ntrain;
nb_c = nb/3;

blcks       = nan(nb_c,nt,3,nsubj);
blck_diffs  = nan(nb_c,nt,3,nsubj);
blck_regime = nan(nb_c,nt,3,nsubj);
cons_regime = nan(nb_c,nt,3,nsubj);
resps       = nan(nb_c,nt,3,nsubj);
rts         = nan(nb_c,nt,3,nsubj);

mu_new   = 55;  % mean of higher distribution
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

a = sig_opti/expe(1).cfg.sgen;      % slope of linear transformation aX+b
b = mu_new - a*expe(1).cfg.mgen;    % intercept of linear transf. aX+b

for isubj = subjlist
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj);
    fprintf('Loading subject %d...\n',isubj);
    load(filename,'expe');
    ib_c = ones(3,1);
    for ib = 1+ntrain:nb+ntrain
        ctype = expe(ib).type;
        switch ctype
            case 'rnd' % random across successive blocks
                ic = 3;
            case 'alt' % always alternating
                ic = 2;
            case 'rep' % always the same
                ic = 1;
        end

        resp_mult = -(expe(ib).resp-1.5)*2;
        blcks(ib_c(ic),:,ic,isubj)          = round(resp_mult.*expe(ib).blck*a+b);
        
        % Organizing difference values between consequent feedback values
        blck_diffs(ib_c(ic),1,ic,isubj)     = 0;
        blck_diffs(ib_c(ic),2:end,ic,isubj) = blcks(ib_c(ic),2:end,ic,isubj) - blcks(ib_c(ic),1:end-1,ic,isubj);
        
        % Labeling each feedback difference within the regime of change 
        blck_regime(ib_c(ic),1,ic,isubj) = 0;
        for it = 2:nt
            if blcks(ib_c(ic),it,ic,isubj) < 50 && blcks(ib_c(ic),it-1,ic,isubj) < 50
                % negative regime
                blck_regime(ib_c(ic),it,ic,isubj) = 1;
            elseif blcks(ib_c(ic),it,ic,isubj) >= 50 && blcks(ib_c(ic),it-1,ic,isubj) >= 50
                % positive regime
                blck_regime(ib_c(ic),it,ic,isubj) = 2;
            elseif blcks(ib_c(ic),it,ic,isubj) >= 50 && blcks(ib_c(ic),it-1,ic,isubj) < 50
                % negative to positive regime
                blck_regime(ib_c(ic),it,ic,isubj) = 3;
            elseif blcks(ib_c(ic),it,ic,isubj) < 50 && blcks(ib_c(ic),it-1,ic,isubj) >= 50
                % positive to negative regime
                blck_regime(ib_c(ic),it,ic,isubj) = 4;
            end
        end
        
        resps(ib_c(ic),:,ic,isubj) = -(expe(ib).resp-2);
        
        % Labeling trials as consistent or inconsistent based on below
        % if fb > 0 & resp ==1 || fb < 0 & resp == 0
        %   consistent 
        % if fb < 0 & resp ==1 || fb > 0 & resp == 0
        %   inconsistent
        for it = 2:nt
            if (blcks(ib_c(ic),it-1,ic,isubj) >= 50 && resps(ib_c(ic),it,ic,isubj) == 1) || ...
               (blcks(ib_c(ic),it-1,ic,isubj) < 50 && resps(ib_c(ic),it,ic,isubj) == 0)
                cons_regime(ib_c(ic),it,ic,isubj) = 1;
            elseif (blcks(ib_c(ic),it-1,ic,isubj) < 50 && resps(ib_c(ic),it,ic,isubj) == 1) || ...
                   (blcks(ib_c(ic),it-1,ic,isubj) >= 50 && resps(ib_c(ic),it,ic,isubj) == 0)
                cons_regime(ib_c(ic),it,ic,isubj) = 2;
            end
        end
        
        rts(ib_c(ic),:,ic,isubj) = expe(ib).rt;
        ib_c(ic) = ib_c(ic)+1;
    end
end

%% Data sorting

ind_subj_quarterly_acc_mean = zeros(numel(subjlist),nt,3,4); % isubj, it, ic, iquarter
ind_subj_quarterly_rt_mean  = zeros(numel(subjlist),nt,3,4); % isubj, it, ic, iquarter
quarterly_acc_mean  = zeros(4,nt,3); % iquarter, it, ic
quarterly_rt_mean   = zeros(4,nt,3); % iquarter, it, ic
quarterly_acc_sem   = zeros(4,nt,3); % iquarter, it, ic
quarterly_rt_sem    = zeros(4,nt,3); % iquarter, it, ic

% filter by quarter
for iq = 1:4
    for ic = 1:3
        blockindex = 4*(iq-1)+1:4*iq;
        for is = 1:numel(subjlist)
            ind_subj_quarterly_acc_mean(is,:,ic,iq) = mean(resps(blockindex,:,ic,subjlist(is)));
            ind_subj_quarterly_rt_mean(is,:,ic,iq)  = mean(rts(blockindex,:,ic,subjlist(is)));
        end
        quarterly_acc_mean(iq,:,ic) = mean(ind_subj_quarterly_acc_mean(:,:,ic,iq));
        quarterly_rt_mean(iq,:,ic)  = mean(ind_subj_quarterly_rt_mean(:,:,ic,iq));
        quarterly_acc_sem(iq,:,ic) = std(ind_subj_quarterly_acc_mean(:,:,ic,iq))/sqrt(numel(subjlist));
        quarterly_rt_sem(iq,:,ic)  = std(ind_subj_quarterly_rt_mean(:,:,ic,iq))/sqrt(numel(subjlist));
    end
end

% Get subject accuracies by quarter and condition
subj_quarterly_acc_mean = mean(ind_subj_quarterly_acc_mean, 2);

% Reaction times

% Organize reaction time based on difference between the current feedback and the previous one
diff_neg_rts = nan(nb_c,nt-2,3,numel(subjlist));
diff_pos_rts = nan(nb_c,nt-2,3,numel(subjlist));

mean_rts_diff_neg = nan(1,1,3,numel(subjlist));
mean_rts_diff_pos = nan(1,1,3,numel(subjlist));
for ic = 1:3
    for isubj = subjlist
        diff_neg_rts(:,:,ic,isubj) = double(bsxfun(@lt,blck_diffs(:,2:end-1,ic,isubj),0)).*rts(:,3:end,ic,isubj);
        diff_pos_rts(:,:,ic,isubj) = double(bsxfun(@ge,blck_diffs(:,2:end-1,ic,isubj),0)).*rts(:,3:end,ic,isubj);
    end
    diff_neg_rts(diff_neg_rts==0) = NaN;
    diff_pos_rts(diff_pos_rts==0) = NaN;
    mean_rts_diff_neg(:,:,ic,:) = mean(mean(diff_neg_rts(:,:,ic,subjlist),'omitnan'));
    mean_rts_diff_pos(:,:,ic,:) = mean(mean(diff_pos_rts(:,:,ic,subjlist),'omitnan'));
end

% Further organize by blck_regime
%                       1: increase/decrease within positive values, 
%                       2: increase/decrease within negative values,
%                       3: increase from negative to positive,
%                       4: decrease from positive to negative
pos_reg_rts = nan(nb_c,nt-2,3,numel(subjlist));
neg_reg_rts = nan(nb_c,nt-2,3,numel(subjlist));
diff_inc_rts = nan(nb_c,nt-2,3,numel(subjlist));
diff_dec_rts = nan(nb_c,nt-2,3,numel(subjlist));
mean_rts_pos_reg = nan(1,1,3,numel(subjlist));
mean_rts_neg_reg = nan(1,1,3,numel(subjlist));
mean_rts_diff_inc = nan(1,1,3,numel(subjlist));
mean_rts_diff_dec = nan(1,1,3,numel(subjlist));

for ic = 1:3
    for isubj = subjlist
        pos_reg_rts(:,:,ic,isubj)  = double(bsxfun(@eq,blck_regime(:,2:end-1,ic,isubj),1)).*rts(:,3:end,ic,isubj);
        neg_reg_rts(:,:,ic,isubj)  = double(bsxfun(@eq,blck_regime(:,2:end-1,ic,isubj),2)).*rts(:,3:end,ic,isubj);
        diff_inc_rts(:,:,ic,isubj) = double(bsxfun(@eq,blck_regime(:,2:end-1,ic,isubj),3)).*rts(:,3:end,ic,isubj);
        diff_dec_rts(:,:,ic,isubj) = double(bsxfun(@eq,blck_regime(:,2:end-1,ic,isubj),4)).*rts(:,3:end,ic,isubj);
    end
    pos_reg_rts(pos_reg_rts==0)   = NaN;
    neg_reg_rts(neg_reg_rts==0)   = NaN;
    diff_inc_rts(diff_inc_rts==0) = NaN;
    diff_dec_rts(diff_dec_rts==0) = NaN;
    mean_rts_pos_reg(:,:,ic,:)    = mean(mean(pos_reg_rts(:,:,ic,subjlist),'omitnan'),'omitnan');
    mean_rts_neg_reg(:,:,ic,:)    = mean(mean(neg_reg_rts(:,:,ic,subjlist),'omitnan'),'omitnan');
    mean_rts_diff_inc(:,:,ic,:)   = mean(mean(diff_inc_rts(:,:,ic,subjlist),'omitnan'),'omitnan');
    mean_rts_diff_dec(:,:,ic,:)   = mean(mean(diff_dec_rts(:,:,ic,subjlist),'omitnan'),'omitnan');
end

% Organize RTs by consistency with the absolute correct/incorrect shape
rts_con = nan(nb_c,nt-1,3,numel(subjlist));
rts_inc = nan(nb_c,nt-1,3,numel(subjlist));
mean_rts_con = nan(1,1,3,numel(subjlist));
mean_rts_inc = nan(1,1,3,numel(subjlist));
for ic = 1:3
    for isubj = subjlist
        rts_con(:,:,ic,isubj) = double(bsxfun(@eq,cons_regime(:,2:end,ic,isubj),1)).*rts(:,2:end,ic,isubj);
        rts_inc(:,:,ic,isubj) = double(bsxfun(@eq,cons_regime(:,2:end,ic,isubj),2)).*rts(:,2:end,ic,isubj);
    end
    rts_con(rts_con==0)     = NaN;
    rts_inc(rts_inc==0)     = NaN;
    mean_rts_con(:,:,ic,:)  = mean(mean(rts_con(:,:,ic,subjlist),'omitnan'),'omitnan');
    mean_rts_inc(:,:,ic,:)  = mean(mean(rts_inc(:,:,ic,subjlist),'omitnan'),'omitnan');
end

%% Plot: Mean block trajectories by quarters by condition

figure(ifig);
ifig = ifig + 1;
clf;
hold on;
for ic = 1:3
    subplot(1,3,ic);
    for iq = 1:4
        plot(1:nt,quarterly_acc_mean(iq,:,ic),'LineWidth',2,'Color', graded_rgb(ic,iq,4));
        shadedErrorBar(1:nt,quarterly_acc_mean(iq,:,ic),quarterly_acc_sem(iq,:,ic),'lineprops',{'Color',graded_rgb(ic,iq,4)});
        ylim([.2 1]);
        hold on;
    end
end
hold off;

%% Plot: Mean block trajectories by condition by quarter
figure(ifig);
ifig = ifig + 1;
clf;
hold on;
for ic = 1:3
    for iq = 1:4
        subplot(1,4,iq);
        plot(1:nt,quarterly_acc_mean(iq,:,ic),'LineWidth',2,'Color', graded_rgb(ic,iq,4));
        shadedErrorBar(1:nt,quarterly_acc_mean(iq,:,ic),quarterly_acc_sem(iq,:,ic),'lineprops',{'Color',graded_rgb(ic,iq,4)});
        yline(mean(quarterly_acc_mean(iq,:,ic)),'Color',graded_rgb(ic,4,4));
        ylim([.2 1]);
        title(sprintf('Quarter %d\nResponse accuracy',iq));
        hold on;
    end
end
hold off;

%% Plot: Mean trial RT trajectories by quarters by condition
figure(ifig);
ifig = ifig + 1;
clf;
hold on;
for ic = 1:3
    for iq = 1:4
        subplot(1,4,iq);
        plot(1:nt,quarterly_rt_mean(iq,:,ic),'LineWidth',2,'Color', graded_rgb(ic,iq,4));
        shadedErrorBar(1:nt,quarterly_rt_mean(iq,:,ic),quarterly_rt_sem(iq,:,ic),'lineprops',{'Color',graded_rgb(ic,iq,4)});
        ylim([.4 2]);
        title(sprintf('Quarter %d\nRTs',iq));
        hold on;
    end
end
hold off;

%% Plot: Mean trial RT trajectories by condition by quarter
figure(ifig);
ifig = ifig + 1;
clf;
hold on;
for ic = 1:3
    for iq = 1:4
        subplot(1,3,ic);
        plot(1:nt,quarterly_rt_mean(iq,:,ic),'LineWidth',2,'Color', graded_rgb(ic,iq,4));        
        shadedErrorBar(1:nt,quarterly_rt_mean(iq,:,ic),quarterly_rt_sem(iq,:,ic),'lineprops',{'Color',graded_rgb(ic,iq,4)});
        ylim([.4 2]);
        hold on;
    end
end
hold off;

%% Plot: Ensure distribution of differences between consequent feedback values are i.i.d.
figure(ifig);
ifig = ifig + 1;
for ic = 1:3
    subplot(3,1,ic);
    histogram(blck_diffs(:,2:end,ic,subjlist),'BinLimits',[-36,36],'BinMethod','integers')
    if ic == 1
        title('Distribution of feedback value differences across conditions');
        xlabel('rep');
    elseif ic == 2
        xlabel('alt');
    else
        xlabel('rnd');
    end
end

%% Plot: RTs: Overall mean over positive and negative feedback values

figure(ifig);
ifig = ifig + 1;
hold on;
fig = scatter(ones(1,numel(subjlist)),mean(mean_rts_diff_neg,3));
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
mean_rt_diff_neg = mean(mean(mean_rts_diff_neg));
scatter(1,mean_rt_diff_neg,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(1,mean_rt_diff_neg,std(mean(mean_rts_diff_neg))/sqrt(numel(subjlist)),'Color',figcolor);

fig = scatter(ones(1,numel(subjlist))*2,mean(mean_rts_diff_pos,3));
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
mean_rt_diff_pos = mean(mean(mean_rts_diff_pos));
scatter(2,mean_rt_diff_pos,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(2,mean_rt_diff_pos,std(mean(mean_rts_diff_pos))/sqrt(numel(subjlist)),'Color',figcolor);

[~,p] = ttest(mean(mean_rts_diff_pos),mean(mean_rts_diff_neg));
sigloc = max(mean_rt_diff_pos, mean_rt_diff_neg);
sigline = line([1 2], [sigloc+.2 sigloc+.2]);
set(sigline,'Color','k');
text(1.5,sigloc+.23,sigstar(p),'FontSize',15,'HorizontalAlignment','center');

xlim([0 3]);
xticks([1 2]);
xticklabels({'\Delta fb < 0','\Delta fb > 0'});
ylabel('RT (s)');
title('RT means');
hold off;

%% Plot: RTs when feedback values cross 0 (increasing vs decreasing)
figure(ifig);
ifig = ifig + 1;

fig      = scatter(ones(1,numel(subjlist)),mean(mean_rts_diff_inc,3));     % rt when coming from neg -> pos feedback
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
mean_rt_diff_inc = mean(mean(mean_rts_diff_inc));
hold on;
scatter(1,mean_rt_diff_inc,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(1,mean_rt_diff_inc,std(mean(mean_rts_diff_inc))/sqrt(numel(subjlist)),'Color',figcolor);

fig      = scatter(2*ones(1,numel(subjlist)),mean(mean_rts_diff_dec,3));   % rt when coming from pos -> neg feedback
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
mean_rt_diff_dec = mean(mean(mean_rts_diff_dec));
scatter(2,mean_rt_diff_dec,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(2,mean_rt_diff_dec,std(mean(mean_rts_diff_dec))/sqrt(numel(subjlist)),'Color',figcolor);

[~,p] = ttest(mean(mean_rts_diff_inc),mean(mean_rts_diff_dec));
sigloc = max(mean_rt_diff_inc, mean_rt_diff_dec);
sigline = line([1 2], [sigloc+.2 sigloc+.2]);
set(sigline,'Color','k');
text(1.5,sigloc+.24,sigstar(p),'FontSize',15,'HorizontalAlignment','center');

xticks([1 2]);
xticklabels({'\Delta fb < 0','\Delta fb > 0'});
ylabel('RT (s)');
title('RT means when fb values cross 0');
xlim([0 3]);
hold off;

%% Plot: RTs over consistent and inconsistent feedback
consis_rts = mean(mean_rts_con,3);
incons_rts = mean(mean_rts_inc,3);

figure(ifig);
ifig = ifig + 1;
hold on;
fig = scatter(1*ones(1,numel(subjlist)),consis_rts);
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
consis_rt = mean(consis_rts);
scatter(1,consis_rt,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(1,consis_rt,std(consis_rts)/sqrt(numel(subjlist)),'Color',figcolor);

fig = scatter(2*ones(1,numel(subjlist)),incons_rts);
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
incons_rt = mean(incons_rts);
scatter(2,incons_rt,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(2,incons_rt,std(incons_rts)/sqrt(numel(subjlist)),'Color',figcolor);

[~,p] = ttest(consis_rts,incons_rts);
sigline = line([1 2], [incons_rt+.2 incons_rt+.2]);
set(sigline,'Color','k');
text(1.5,incons_rt+.21,sigstar(p),'FontSize',20,'HorizontalAlignment','center');

xticks([1 2]);
xticklabels({'Consistent fb','Inconsistent fb'});
ylabel('RT (s)');
title(sprintf('RT means over Consistent and Inconsistent feedback values'));
xlim([0 3]);
hold off;

%% Plot: RTs over consistent and inconsistent feedback (by condition)

consis_rts_rep = mean(mean_rts_con(:,:,1,:),3);
consis_rts_alt = mean(mean_rts_con(:,:,2,:),3);
incons_rts_rep = mean(mean_rts_inc(:,:,1,:),3);
incons_rts_alt = mean(mean_rts_inc(:,:,2,:),3);

figure(ifig);
ifig = ifig + 1;
hold on;
fig = scatter(.5*ones(1,numel(subjlist)),consis_rts_rep,'s');
fig.MarkerEdgeAlpha = 0.2;
figcolor = [1 0 0];
fig.MarkerEdgeColor = figcolor;
fig = scatter(1*ones(1,numel(subjlist)),incons_rts_rep,'d');
fig.MarkerEdgeAlpha = 0.2;
fig.MarkerEdgeColor = figcolor;
consis_rt_rep = mean(consis_rts_rep);
incons_rt_rep = mean(incons_rts_rep);
scatter(.5,consis_rt_rep,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(.5,consis_rt_rep,std(consis_rts_rep)/sqrt(numel(subjlist)),'Color',figcolor);
scatter(1,incons_rt_rep,'x','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(1,incons_rt_rep,std(incons_rts_rep)/sqrt(numel(subjlist)),'Color',figcolor);

fig = scatter(1.5*ones(1,numel(subjlist)),consis_rts_alt,'s');
fig.MarkerEdgeAlpha = 0.2;
figcolor = [.2 .8 .2];
fig.MarkerEdgeColor = figcolor;
fig = scatter(2*ones(1,numel(subjlist)),incons_rts_alt,'d');
fig.MarkerEdgeAlpha = 0.2;
fig.MarkerEdgeColor = figcolor;
consis_rt_alt = mean(consis_rts_alt);
incons_rt_alt = mean(incons_rts_alt);
scatter(1.5,consis_rt_alt,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(1.5,consis_rt_alt,std(consis_rts_alt)/sqrt(numel(subjlist)),'Color',figcolor);
scatter(2,incons_rt_alt,'x','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(2,incons_rt_alt,std(incons_rts_alt)/sqrt(numel(subjlist)),'Color',figcolor);

[~,p] = ttest(consis_rts_rep,incons_rts_rep);
sigloc = max(consis_rt_rep, incons_rt_rep);
sigloc = sigloc+sigloc/10;
sigline = line([.5 1], [sigloc sigloc]);
set(sigline,'Color','k');
text(.75,sigloc+sigloc/20,sigstar(p),'FontSize',10,'HorizontalAlignment','center');

[~,p] = ttest(consis_rts_alt,incons_rts_alt);
sigloc = max(consis_rt_alt, incons_rt_alt);
sigloc = sigloc+sigloc/10;
sigline = line([1.5 2], [sigloc sigloc]);
set(sigline,'Color','k');
text(1.75,sigloc+sigloc/20,sigstar(p),'FontSize',20,'HorizontalAlignment','center');

xticks([.5 1 1.5 2]);
xticklabels({'Cons. fb (rep)','Inc. fb (rep)', 'Cons. fb (alt)','Inc. fb (alt)'});
ylabel('RT (s)');
title(sprintf('RT means over Consistent and Inconsistent feedback values within conditions'));
xlim([0 2.5]);
hold off;

%% Plot: RTs over confirmatory vs disconfirmatory evidence (of the previous choice)
confirm_pos_rts = mean(mean_rts_pos_reg,3);
confirm_neg_rts = mean(mean_rts_neg_reg,3);
disconf_rts     = mean([mean(mean_rts_diff_inc,3) mean(mean_rts_diff_dec,3)],2);
figure(ifig);
ifig = ifig + 1;
hold on;
fig = scatter(1*ones(1,numel(subjlist)),confirm_pos_rts);
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
confirm_pos_rt = mean(confirm_pos_rts);
scatter(1,confirm_pos_rt,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(1,confirm_pos_rt,std(confirm_pos_rts)/sqrt(numel(subjlist)),'Color',figcolor);

fig = scatter(2*ones(1,numel(subjlist)),confirm_neg_rts);
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
confirm_neg_rt = mean(confirm_neg_rts);
scatter(2,confirm_neg_rt,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(2,confirm_neg_rt,std(confirm_neg_rts)/sqrt(numel(subjlist)),'Color',figcolor);

fig = scatter(3*ones(1,numel(subjlist)),disconf_rts);
fig.MarkerEdgeAlpha = 0.2;
figcolor = get(fig,'CData');
disconf_rt = mean(disconf_rts);
scatter(3,disconf_rt,'d','filled','MarkerEdgeColor',figcolor,'MarkerFaceColor',figcolor);
errorbar(3,disconf_rt,std(disconf_rts)/sqrt(numel(subjlist)),'Color',figcolor);

[~,p] = ttest(confirm_pos_rts,confirm_neg_rts);
sigline = line([1 2], [confirm_pos_rt+.2 confirm_pos_rt+.2]);
set(sigline,'Color','k');
text(1.5,confirm_pos_rt+.21,sigstar(p),'FontSize',20,'HorizontalAlignment','center');

[~,p] = ttest(confirm_pos_rts,disconf_rts);
sigline = line([1 3], [confirm_pos_rt+.24 confirm_pos_rt+.24]);
set(sigline,'Color','k');
text((1+3)/2,confirm_pos_rt+.26,sigstar(p),'FontSize',20,'HorizontalAlignment','center');

[~, p] = ttest(confirm_neg_rts,disconf_rts);
sigline = line([2 3], [confirm_pos_rt+.18 confirm_pos_rt+.18]);
set(sigline,'Color','k');
text(2.5,confirm_pos_rt+.2,sigstar(p),'FontSize',15,'HorizontalAlignment','center');

xticks([1 2 3]);
xticklabels({'Confirmatory pos.','Confirmatory neg.','Disconfirmatory'});
ylabel('RT (s)');
title(sprintf('RT means over different regimes of feedback values\n (Confirmatory vs. Disconfirmatory evidence)'));
xlim([0 4]);

%% Local functions
function rgb = rgb_ic(ic)
    switch ic
        case 3 % random across successive blocks
            rgb = [1 0 0];
        case 2 % always alternating
            rgb = [0 1 0];
        case 1 % always repeating
            rgb = [0 0 1];
    end
end

function rgb = graded_rgb(ic,ib,nb)
xc = linspace(.8,.2,nb);

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
        stars = 'n.s.';
    end
end




