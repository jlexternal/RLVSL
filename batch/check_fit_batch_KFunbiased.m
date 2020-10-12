% check_fit_batch_KFunbiased
%
% Objective: Check fitted parameters for the unbiased KF model on the random/novel
%             condition. Simulate models on the repeating and alternating conditions 
%             with the fitted parameters and identify blocks that are easy.
%
% Version:   Code for fits from the fit_noisyKF_epsibias.m fitting code (pre 12 Oct 2020)
%
% Jun Seok Lee <jlexternal@gmail.com>

clear all
clc

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);
addpath('./param_fit_PF_KFunbiased')
load('subj_resp_rew_all.mat'); % load all relevant subject data

% fitting options
cscheme = 'ths';
lscheme = 'sym';
nscheme = 'upd';
icond = 3;

params = nan(3,nsubjtot); % kini,kinf,zeta 

% load and gather all the fit data
out_fit_all = cell(1,nsubjtot);
for isubj = subjlist
    filename = sprintf('./param_fit_PF_KFunbiased/options_fit/out_fit_KFunbiased_%s_%s_%s_%d_%02d.mat',...
                       cscheme,lscheme,nscheme,nsubj,isubj);
    load(filename);
    out_fit_all{icond,isubj} = out_fit{icond,isubj};
    
    params(1,isubj) = out_fit{icond,isubj}.kini;
    params(2,isubj) = out_fit{icond,isubj}.kinf;
    params(3,isubj) = out_fit{icond,isubj}.zeta;
end

out_fit_rnd = out_fit_all(3,:);
%save(sprintf('out_fit_rnd_unbiasedKF_%s_%s_%s',cscheme,lscheme,nscheme),'out_fit_rnd');
%% Plot fitted parameters
figure
hold on
for i=1:size(params,1)
	scatter(i*ones(1,nsubj)+normrnd(0,.05,[1,nsubj]),params(i,~isnan(params(i,:))),80,'filled','MarkerFaceAlpha',.1);
    colors = get(gca,'ColorOrder');
    yline(mean(params(i,:),'omitnan'),'LineWidth',2,'Color',colors(i,:));
    set(gca,'ColorOrderIndex',i);
    errorbar(i,mean(params(i,:),'omitnan'),std(params(i,:),'omitnan'),'LineWidth',2);
end
yline(0,':');
yline(1,':');
hold off;
xlim([0 4]);
xticks([1:3]);
set(gca,'xticklabel',{'kini','kinf','zeta'},'FontSize',16);

%% Simulate models with fitted parameters

addpath('../');

% Experimental parameters
cfg.nb = 16; % number of blocks
cfg.nt = 16; % number of trials
% Generative parameters of winning distribution with FNR of 25%
cfg.ms = .55; % sampling mean
cfg.vs = .07413^2; % sampling variance
% Assumptions of the model
cfg.sbias_cor = false;
cfg.sbias_ini = false;
cfg.cscheme = cscheme;
cfg.lscheme = lscheme;
cfg.nscheme = nscheme;
% Model parameters
cfg.ns      = 1000; % Number of simulated agents to generate per given parameter
cfg.epsi    = 0;
cfg.ksi     = 0;
cfg.theta   = 0;
% Simulation settings
cfg.sameexpe = true;    % true if all sims see the same reward scheme

nb = cfg.nb;
nt = cfg.nt;
nc = 3;

% simulate experiment for each subject+condition the unbiased KF model w/ fitted parameters
sim_out = cell(nsubjtot,3);
for isubj = subjlist
    fprintf('Simulating model on subject %d\n',isubj);
    cfg.kini = params(1,isubj);
    cfg.kinf = params(2,isubj);
    cfg.zeta = params(3,isubj);
    for icond = 1:3
        cfg.firstresp = subj_resp_rew_all(isubj).resp(:,1,icond); % simulations make the same 1st choice as subject
        cfg.compexpe  = subj_resp_rew_all(isubj).rew_expe(:,:,icond)/100;
        sim_out{isubj,icond} = sim_epsibias_fn(cfg);
    end
end

%% Percentage of simulations with absolute repeat blocks
abs_rep_blks_sims = nan(nb,nc,nsubjtot);
abs_rep_blks_subj = nan(nb,nc,nsubjtot);

errTol = 1; % error tolerance (number of allowed mistakes)

for isubj = subjlist
    for icond = 1:3
        abs_rep_blks_sims(:,icond,isubj) = mean(sum(sim_out{isubj,icond}.resp,2) <= nt+errTol,3);
        abs_rep_blks_subj(:,icond,isubj) = sum(subj_resp_rew_all(isubj).resp(:,:,icond),2) <= nt+errTol;
    end
end

mean_abs_rep_sims = squeeze(mean(abs_rep_blks_sims,1));
mean_abs_rep_subj = squeeze(mean(abs_rep_blks_subj,1));

%plotting
mean_vals = [mean(mean_abs_rep_subj,2,'omitnan') mean(mean_abs_rep_sims,2,'omitnan')];
err_vals  = [std(mean_abs_rep_subj,1,2,'omitnan') std(mean_abs_rep_sims,1,2,'omitnan')]/sqrt(nsubj);
b = bar(mean_vals);
hold on;
for i = 1:2
    errorbar(b(i).XData+b(i).XOffset,mean_vals(:,i),err_vals(:,i),'k','LineStyle','none');
end
xticklabels({'Repeating','Alternating','Novel'});
xlabel('Experimental condition')
legend({'Subjects','Simulations'})
title(sprintf('Mean percentage of absolute repeating correct response blocks\nerror bars SEM'));
hold off

%% Calculate model-free measures 
% 1/ Learning curves
% 2/ p(correct)
% 3/ p(repeat 1st response)
% 4/ p(repeat previous response)


lcurve= nan(4,nt,nc,nsubj,2); % quarter, trials, condition, subject, source

p_cor = nan(nb,2,nc,nsubj);
p_1st = nan(nb,2,nc,nsubj);
p_rep = nan(nb,2,nc,nsubj);

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    
    % subjects
    resp = subj_resp_rew_all(isubj).resp;
    resp(resp==2)=0;
    for iq = 1:4
        blockrange = 4*(iq-1)+1:4*(iq-1)+4;
        lcurve(iq,:,:,jsubj,1) = mean(resp(blockrange,:,:),1);
    end
    p_cor(:,1,:,jsubj) = sum(resp,2)/nt;
    p_1st(:,1,:,jsubj) = sum(bsxfun(@eq,resp(:,1,:),resp(:,2:end,:)),2)/(nt-1);
    p_rep(:,1,:,jsubj) = sum(bsxfun(@eq,resp(:,1:nt-1,:),resp(:,2:nt,:)),2)/(nt-1);
    
    % simulations
    for ic = 1:nc
        resp = sim_out{isubj,ic}.resp;
        resp(resp==2) = 0;
        for iq = 1:4
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            lcurve(iq,:,ic,jsubj,2) = mean(mean(resp(blockrange,:,:),1),3);
        end
        p_cor(:,2,ic,jsubj) = mean(sum(resp,2)/nt,3);
        p_1st(:,2,ic,jsubj) = mean(sum(bsxfun(@eq,resp(:,1,:),resp(:,2:end,:)),2)/(nt-1),3);
        p_rep(:,2,ic,jsubj) = mean(sum(bsxfun(@eq,resp(:,1:nt-1,:),resp(:,2:nt,:)),2)/(nt-1),3);
    end
end

%% plot 2-D histograms
ic_plot = 3; % condition
iq_plot = 3; % quarter
blockrange = 4*(iq_plot-1)+1:4*(iq_plot-1)+4;
sourceStr = {'Subjects','Simulations'};

figure;
for isource = 1:2
    subplot(2,3,3*(isource-1)+1);
    hist3([reshape(p_1st(blockrange,isource,ic_plot,:),[numel(p_1st(blockrange,isource,ic_plot,:)) 1]) ...
           reshape(p_cor(blockrange,isource,ic_plot,:),[numel(p_cor(blockrange,isource,ic_plot,:)) 1])],...
            'Edges',{0:.1:1 0:.1:1},'CDataMode','auto');
    xlabel('p(1st response)');
    ylabel('p(correct)');
    title(sprintf('%s\ncorrect x 1st response',sourceStr{isource}));

    subplot(2,3,3*(isource-1)+2);
    hist3([reshape(p_1st(blockrange,isource,ic_plot,:),[numel(p_1st(blockrange,isource,ic_plot,:)) 1]) ...
           reshape(p_rep(blockrange,isource,ic_plot,:),[numel(p_rep(blockrange,isource,ic_plot,:)) 1])],...
            'Edges',{0:.1:1 0:.1:1},'CDataMode','auto');
    xlabel('p(1st response)');
    ylabel('p(repeat)');
    title(sprintf('%s\nrepeat x 1st response',sourceStr{isource}));

    subplot(2,3,3*(isource-1)+3);
    hist3([reshape(p_cor(blockrange,isource,ic_plot,:),[numel(p_cor(blockrange,isource,ic_plot,:)) 1]) ...
           reshape(p_rep(blockrange,isource,ic_plot,:),[numel(p_rep(blockrange,isource,ic_plot,:)) 1])],...
            'Edges',{0:.1:1 0:.1:1},'CDataMode','auto');
    xlabel('p(correct)');
    ylabel('p(repeat)');
    title(sprintf('%s\nrepeat x correct',sourceStr{isource}));
end
sgtitle(sprintf('Condition: %d\nQuarter: %d',ic_plot,iq_plot))

%% calculate learning curves across the quarters
addpath('../')

figure
for ic = 1:nc
    legtxt = cell(8,1);
    subplot(1,3,ic)
    hold on
    for iq = 1:4
        errorbar(1:16,mean(lcurve(iq,:,ic,:,1),4),std(lcurve(iq,:,ic,:,1),1,4)/sqrt(nsubj),...
            'CapSize',0,'Color',graded_rgb(ic,iq),'LineWidth',1.5)
        legtxt{2*(iq-1)+1} = sprintf('Subj Q%d',iq);
        shadedErrorBar(1:16,mean(lcurve(iq,:,ic,:,2),4),std(lcurve(iq,:,ic,:,2),1,4)/sqrt(nsubj),...
            'lineprops',{'-.','Color',graded_rgb(ic,iq),'LineWidth',1.5},'patchSaturation',.1)
        legtxt{2*(iq-1)+2} = sprintf('Model Q%d',iq);
    end
    ylim([.4 1]);
    xticks(4:4:16)
    legend(legtxt,'Location','southeast')
    title(sprintf('Condition: %d',ic))
    hold off
end
sgtitle(sprintf('Learning curves\nError bars SEM'))

%% Local functions
function rgb = graded_rgb(ic,iq)
           
    red = [1.0 .92 .92; .98 .80 .80; .97 .64 .64; .96 .49 .49];
    gre = [.94 .99 .94; .85 .95 .83; .74 .91 .70; .63 .87 .58];
    blu = [.93 .94 .98; .78 .85 .94; .61 .74 .89; .44 .63 .84];
           
    rgb = cat(3,red,gre);
    rgb = cat(3,rgb,blu);
    
    rgb = rgb(iq,:,ic);
end