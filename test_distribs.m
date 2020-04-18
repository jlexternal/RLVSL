% Code to test statistics and dynamics of generated values for experiment RLvSL
%
% Author:   Jun LEE
% Date:     November 2019

%% generate experiments
n_gens  = 100;  % number of observers

% use gen_expe to generate setup data
for i = 1:n_gens
    test(i).expe = gen_expe_rlvsl(i);
end

%% plot optimal KF trajectories on multiple blocks

nb  = numel(test(1).expe);
nt  = numel(test(1).expe(1).blck);

% consolidate outcomes from all blocks into a single matrix
obs = zeros(n_gens*nb,nt);
obsCtr = 1;
for igen = 1:n_gens
    for i = 1:nb
        obs(obsCtr,:) = test(igen).expe(i).blck; 
        obsCtr = obsCtr + 1;
    end
end

% transform distribution to the real one seen by participants
mu_new  = 10;   % desired mean of distribution
sig_new = 15;   % desired std of distribution
a = sig_new/test(1).expe(1).cfg.sgen;       % slope of linear transformation aX+b
b = mu_new - a*test(1).expe(1).cfg.mgen;    % intercept of linear transf. aX+b
% apply transformation
obs = round(obs*a+b);

nk = 100; % number of kalman filter optimal trackers
alpha   = .2;   % learning rate
zeta    = 1;    % learning noise
r_sd    = 15;   % standard deviation of reward distribution
vs      = r_sd^2; % sampling variance (set equal to true variance of distribution)

ests = zeros(size(obs,1),nt,nk);
vars = zeros(size(obs,1),nt,nk);

% calculate optimal KF tracking
for ib = 1:size(obs,1)
    vars(ib,1,:) = 1e6; % initialize flat posterior variance
    for it = 2:nt
        kt              = vars(ib,it-1,:)./(vars(ib,it-1,:)+vs); % kalman gain
        ests(ib,it,:)   = ests(ib,it-1,:)+(obs(ib,it-1,:)-ests(ib,it-1,:)).*kt.*(1+randn(1,1,nk)*zeta);
        vars(ib,it,:)   = (1-kt).*vars(ib,it-1,:);
    end
end
%% Plot optimal KF trajectories


blockrange = 100:200; % choose the range of blocks you want to consider (this is kind of arbitrary) for marginally faster calculation
xc = linspace(1,0.2,numel(blockrange));
figure(1);
yline(10,'LineWidth',2); % true generative difference mean
hold on;
est_means = mean(ests(blockrange,:,:),3); 
for i = 1:length(blockrange)
    rgb = [xc(i) 0 1-xc(i)];
    plot(1:16,est_means(i,:),'Color',rgb);
    hold on;
    pause(.1);
end
xlabel('trial number');
ylabel('tracking estimate');
hold off;
fprintf('The standard deviation of estimates at trial 10 is %f \n',std(est_means(:,10)));


%% Plot correlation matrix
figure(2);
imagesc(corr(obs));
%colormap(gray);
xlabel('trial i');
ylabel('trial j');



