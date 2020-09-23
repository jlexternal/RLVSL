% analyze_sim_pibias
clear all
clc

cfg = struct;
% Experimental parameters
cfg.nb = 8;     % number of blocks
cfg.nt = 16;    % number of trials
% Generative parameters of winning distribution with FNR of 25%
cfg.ms = .55;       % sampling mean
cfg.vs = .07413^2;  % sampling variance
% Assumptions of the model
cfg.lscheme     = 'sym'; % 'ind'-independent action values;  'sym'-symmetric action values
cfg.nscheme     = 'upd'; % 'rpe'-noise scaling w/ RPE;       'upd'-noise scaling w/ value update
% Model parameters
cfg.ns      = 28;   % Number of simulated agents to generate per given parameter
cfg.pi      = 0.8;  % Probability of the block governed by structure learning (repetition)
cfg.alpha   = 0.3; 	% Constant learning rate
cfg.zeta    = 0.4; 	% Learning noise scaling parameter
cfg.ksi     = 0.0; 	% Learning noise constant parameter
cfg.theta   = 0.0; 	% Softmax temperature
% Simulation settings
cfg.sameexpe = false;   % true if all sims see the same reward scheme

%% 1/ Simulate trial-based and block-based bias responses for given parameters

mean_p = [0.4196 0.5536 0.7321 0.7500];
sem_p = [0.0490 0.0623 0.0549 0.0399];

ipi = 1;
% testing multiple pi parameters based on subjects mean repetition proportion and SEM
pis = normrnd(mean_p(ipi),sem_p(ipi),[cfg.ns 1]);

sim_blk_test = {};
sim_trl_test = {};

%% Testing model sims with parameters spread around a mean

sim_blk = struct;
sim_trl = struct;
ns = cfg.ns;
for bscheme = {'blk','trl'}
    for is = 1:ns
        cfg.pi = pis(is);
        cfg.ns = 1;
        cfg.bscheme = bscheme{1}; % 'trl'-trial-based structure bias    'blk'-block-based structure bias
        if strcmpi(bscheme,'blk')
            sim_blk_test{is} = sim_pibias_fn(cfg);
            sim_blk.resp(:,:,is) = sim_blk_test{is}.resp;
        else
            sim_trl_test{is} = sim_pibias_fn(cfg);
            sim_trl.resp(:,:,is) = sim_trl_test{is}.resp;
        end
    end
end
nb = cfg.nb;
nt = cfg.nt;
ns = 28;


%% Generate model sims with single set of parameters
for bscheme = {'blk','trl'}
    cfg.bscheme = bscheme{1}; % 'trl'-trial-based structure bias    'blk'-block-based structure bias
    if strcmpi(bscheme,'blk')
        sim_blk = sim_pibias_fn(cfg);
        
    else
        sim_trl = sim_pibias_fn(cfg);
        
    end
end
nb = cfg.nb;
nt = cfg.nt;
ns = cfg.ns;

%% 2/ Generate model-free proportion heatmaps for the two bias type models
p_cor = nan(nb,2,ns); % blocks, bias type, number of agents
p_1st = nan(nb,2,ns);
p_rep = nan(nb,2,ns);

idx_rep = nan(nb,2,ns);

for itype = 1:2 % 1:block, 2:trial
    for is = 1:ns
        for ib = 1:nb
            % load sim responses
            if itype == 1
                resp = sim_blk.resp(ib,:,is);
            else
                resp = sim_trl.resp(ib,:,is);
            end
            resp(resp==2)=0;
            
            % check for absolute repetition during the block
            idx_rep(ib,itype,is) = isequal(ones(1,nt)*resp(1),resp);

            p_cor(ib,itype,is) = sum(resp)/nt;
            p_1st(ib,itype,is) = numel(resp(resp == resp(1)))/nt;
            p_rep(ib,itype,is) = sum(bsxfun(@eq,resp(1:nt-1),resp(2:nt)))/nt;
        end
    end
end

% plot
figure;
nbins = 10;
biasstr = {'Block-based bias','Trial-based bias'};
for itype = 1:2
    test1 = [];
    test2 = [];
    test3 = [];
    for is = 1:ns
        test1 = cat(1,test1,p_1st(:,itype,is));
        test2 = cat(1,test2,p_cor(:,itype,is));
        test3 = cat(1,test3,p_rep(:,itype,is));
    end
    subplot(2,3,3*(itype-1)+1);
    histogram2(test1, test2,[nbins nbins],'Normalization','probability','ShowEmptyBins','on','FaceColor','flat');
    xlabel('p(1st response)');
    ylabel('p(correct)');
    title(sprintf('%s: correct x 1st response',biasstr{itype}));

    subplot(2,3,3*(itype-1)+2);
    histogram2(test1, test3,[nbins nbins],'Normalization','probability','ShowEmptyBins','on','FaceColor','flat');
    xlabel('p(1st response)');
    ylabel('p(repeat)');
    title(sprintf('%s: repeat x 1st response',biasstr{itype}));

    subplot(2,3,3*(itype-1)+3);
    histogram2(test2, test3,[nbins nbins],'Normalization','probability','ShowEmptyBins','on','FaceColor','flat');
    xlabel('p(correct)');
    ylabel('p(repeat)');
    title(sprintf('%s: repeat x correct',biasstr{itype}));
end

% 3/ Calculate percentages of total repetition given quarter given condition
rep_percs = nan(2,ns);
for is = 1:ns
    for itype = 1:2
        rep_percs(itype,is) = sum(idx_rep(:,itype,is))/nb;
    end
end

% plot
figure;
rep_percs_m = mean(rep_percs,2);
hold on;
bplot = bar(rep_percs_m);
er = std(rep_percs,1,2)/sqrt(ns);
x = [1:2];
errorbar(x, rep_percs_m, er, 'k', 'linestyle', 'none');
xticks([1 2]);


%yline(cfg.pi);
yline(mean(pis));
xticklabels({'Block-based','Trial-based'});
xlabel('Bias type');
ylabel('Proportion');
title(sprintf('Proportion of absolute repetitive choice blocks\nError bars SEM'));
hold off
