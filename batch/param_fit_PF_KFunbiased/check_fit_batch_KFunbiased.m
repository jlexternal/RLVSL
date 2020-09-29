% check_fit_batch_KFunbiased
%
% Objective: Check fitted parameters for the unbiased KF model on the random/novel
%             condition. Simulate models on the repeating and alternating conditions 
%             with the fitted parameters and identify blocks that are easy.

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);
load('subj_resp_rew_all.mat'); % load all relevant subject data

% fitting options
cscheme = 'qvs';
lscheme = 'sym';
nscheme = 'rpe';
icond = 3;

params = nan(3,nsubjtot); % kini,kinf,zeta 

% load and gather all the fit data
out_fit_all = cell(1,nsubjtot);
for isubj = subjlist
    filename = sprintf('./transferred/out_fit_KFunbiased_%s_%s_%s_%d_%02d.mat',cscheme,lscheme,nscheme,nsubj,isubj);
    load(filename);
    out_fit_all{icond,isubj} = out_fit{icond,isubj};
    
    params(1,isubj) = out_fit{icond,isubj}.kini;
    params(2,isubj) = out_fit{icond,isubj}.kinf;
    params(3,isubj) = out_fit{icond,isubj}.zeta;
end

out_fit_rnd = out_fit_all(3,:);
save(sprintf('out_fit_rnd_unbiasedKF_%s_%s_%s',cscheme,lscheme,nscheme),'out_fit_rnd');
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
hold off;
xlim([0 4]);
ylim([0 1]);
xticks([1:3]);
xticklabels({'kini','kinf','zeta'});

%% Simulate models with fitted parameters

addpath('../../');

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

% simulate experiment for each subject+condition the unbiased KF model w/ fitted parameters
sim_out = cell(nsubjtot,2);
for isubj = subjlist
    fprintf('Simulating model on subject %d\n',isubj);
    cfg.kini = params(1,isubj);
    cfg.kinf = params(2,isubj);
    cfg.zeta = params(3,isubj);
    for icond = 1:2
        cfg.firstresp = subj_resp_rew_all(isubj).resp(:,1,icond); % simulations make the same 1st choice as subject
        cfg.compexpe  = subj_resp_rew_all(isubj).rew_expe(:,:,icond)/100;
        sim_out{isubj,icond} = sim_epsibias_fn(cfg);
    end
end

%% Percentage of simulations with absolute repeat blocks
nt = 16;
abs_rep_blks_sims = nan(16,2,nsubjtot);
abs_rep_blks_subj = nan(16,2,nsubjtot);

errTol = 1; % error tolerance (number of allowed mistakes)

for isubj = subjlist
    for icond = 1:2
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
b.XOffset
hold on;
for i = 1:2
    errorbar(b(i).XData+b(i).XOffset,mean_vals(:,i),err_vals(:,i),'k','LineStyle','none');
end
xticklabels({'Repeating','Alternating'});
legend({'Subjects','Simulations'})
title(sprintf('Mean percentage of absolute repeating correct response blocks\nerror bars SEM'));
hold off

%% Calculate model-free proportion curves
% 1/ Learning curve
% 2/ Repeat 1st response
% 3/ Repeat previous response















