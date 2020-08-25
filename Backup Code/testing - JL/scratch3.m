% This scratch code has a single filter for both options, tracking them together via
% perceived value differences

% This code does not consider a model of pattern learning

% Jun Seok Lee - Oct 2019

%% Generative distribution parameters
clear all;
close all;
mu_vd       = 0.1; % mean of the true value difference between options
sig_vd      = 0.1; % standard deviation of value difference distributions

%% Generate blocks and simulate them on artificial observers

% Declare observer variables 
choices = []; % choices (sign of estimate; +1 or -1)
ests    = []; % estimations (value of estimated mean)
k       = []; % learning rates
obs     = []; % observations

nsimblocks = 10000;  % number of different blocks to generate
ntrials = 10;       % number of trials per block
nsims   = 1000;    % number of simulations on a given block structure
nblocks = nsimblocks; % number of blocks to generate before the good ones are chosen

% Generate blocks
blocks = normrnd(mu_vd,sig_vd,nblocks,ntrials); 

runblocks_end   = false;     
%% (Rerun this part after good blocks chosen below)
runblocks = true;
while runblocks
    % Initial filter and estimation variables
    nu          = 0;            % init process uncertainty
    omega       = sig_vd.^2;    % observation noise
    zeta        = 1;            % learning noise
    ests        = zeros(nsims,ntrials+1,nblocks);	% init estimation of choice 1 at 50 (flat prior)
    k           = ones(nsims,ntrials+1,nblocks);	% initial learning rate
    if ~runblocks_end
        disp('generating 1st set of blocks (to be sorted)');
    else
        disp('running blocks chosen after rank sorting');
    end
    for iblock = 1:nblocks
        w = 3.^2; % initialized posterior variance for choice estimate 1
        k(:,1,iblock) = lrate(w, nu, omega);

        for itrial = 1:ntrials
            % Choice step (argmax)
            if itrial == 1
                choices(:,1,iblock)      = datasample([1 -1],nsims);
            else
                choices(:,itrial,iblock) = choices(:,itrial-1,iblock).*decision(ests(:,itrial,iblock)); % if estimate is neg, change choice 
            end
            % Show outcome of choice: outcome sampling step
            obs(:,itrial,iblock) = blocks(iblock,itrial).*choices(:,itrial,iblock);

            % Make a new estimate of mean: estimation step
            ests(:,itrial+1,iblock) = kalman(ests(:,itrial,iblock),k(:,itrial,iblock),obs(:,itrial,iblock),zeta);

            % Update learning rate: filtering step
            k(:,itrial+1,iblock)	= lrate(w, nu, omega);
            w = (1-k(:,itrial+1,iblock)).*(w+nu);
        end
    end

    if runblocks_end
        break;
    end

    % Choose blocks that are good

    % Set cutoff ending accuracy interval
    acc_val     = .8;
    acc_prec    = .01; 
    nblocks     = 10; % # of final desired blocks

    % find subset of blocks whose final accuracies (proportion of correct sign of estimate) lie within cutoff interval
    mean_blocks = mean(bsxfun(@eq,sign(ests),1),1);
    ind_blocks  = find(mean_blocks(:,11,:)>acc_val-acc_prec & mean_blocks(:,11,:)<acc_val+acc_prec);
    acc_blocks  = blocks(ind_blocks,:);
    fprintf('%d blocks found within acceptable interval', numel(acc_blocks));
    % rank them by distance away from mean average and mean variance
    acc_ests = ests(:,[2:end],ind_blocks);

    % need q values to get mean / var rankings from optimal observer
    q_means = mean(acc_ests,2);
    q_vars  = var(acc_ests,0,2);
    %disp(mean(q_means)); %debug
    %disp(mean(q_vars)); %debug
    diff_means_sq   = (mean(q_means,1) - mean(mean(q_means,1))).^2;   % calculate squared diff. of mean q per block to overall mean q
    diff_vars_sq    = (mean(q_vars,1)  - mean(mean(q_vars,1))).^2;     % calculate squared diff. of var q per block to overall mean var
    % rank these two measures
    [~,imeans]  = sort(diff_means_sq,'ascend');     % i___ - 'i' refers to index
    [~,ivars]   = sort(diff_vars_sq,'ascend');
    rank_means  = 1:length(diff_means_sq);
    rank_vars   = 1:length(diff_vars_sq);
    rank_means(imeans)  = rank_means; 
    rank_vars(ivars)    = rank_vars;
    % add the ranks (lower ranks will be chosen as test blocks)
    rank_total = rank_means + rank_vars;

    [~,iblocks] = sort(rank_total,'ascend'); % 'iblocks' contains indices of blocks with best ranking
    % choose the top 10 most average blocks
    iblocks = iblocks(1:nblocks);
    blocks = acc_blocks(iblocks,:);

    runblocks_end = true;
end

%% plots

A = mean(bsxfun(@eq,sign(ests),1),1);
B = mean(ests,1);
figure(1);
subplot(2,2,1);
hold on;
for i = 1:nblocks
    plot([1:ntrials],A(:,[2:end],i));
end
yline(.8,'Color',[0 0 0],'LineWidth',2);
title(sprintf('Accuracy per a given block \n for %d simulated observers',nsims));
subplot(2,2,2);
hold on;
for j = 1:nblocks
    plot([1:ntrials],B(:,[2:end],j));
end
yline(mu_vd,'Color',[0 0 0],'LineWidth',2);
title('Estimated means');
subplot(2,2,3);
hold on;
for i = 1:nblocks
    plot([1:ntrials],blocks(i,[1:end]));
end
yline(mu_vd,'Color',[0 0 0],'LineWidth',2);
title(sprintf('Observations (value differences \n if correct option chosen everytime)'));
hold off;
%{
figure(1);
subplot(2,1,1);
scatter([1:ntrials],x(1,:),'MarkerEdgeColor',[1 0 0]); % observations bandit 1
hold on;
scatter([1:ntrials],x(2,:),'MarkerEdgeColor',[0 0 1]); % observations bandit 2
plot([1:ntrials+1]-1, ests(1,:), 'Color', [.5 0 0]); % estimations bandit 1
plot([1:ntrials+1]-1, ests(2,:), 'Color', [0 0 .5]); % estimations bandit 2
scatter([1:ntrials]-1, choicepts,'x','MarkerEdgeColor',[0 0 0],'LineWidth',2);
yline(50-d,'--','Color',[.75 0 0]);
yline(50+d,'--','Color',[0 0 .75]);
ylim([40 60]);
title('Bandit, tracking, choice dynamics');
hold off;

subplot(2,1,2);
scatter([1:ntrials+1]-1,k(1,:),'o','MarkerFaceColor',[.5 0 0],'MarkerEdgeColor', [.5 0 0]);
hold on;
scatter([1:ntrials+1]-1,k(2,:),'o','MarkerFaceColor',[0 0 .5],'MarkerEdgeColor', [0 0 .5]);
title('Learning rates');
hold off;
%}

% Local functions
function out = kalman(x,k,o,zeta) %(previous estimate, kalman gain, observation)
    d1 = size(x,1);
    d2 = size(x,2);
    out = x+k.*(o-x).*(1+randn(d1,d2).*zeta);
end

function out = lrate(w,nu,omega)
    out = (w+nu)./(w+nu+omega);
end

function out = decision(estimate)
    out = eq(sign(estimate),1);
    out = double(out);
    out(out==0) = -1;
end

