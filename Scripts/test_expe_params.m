%% Determine bandit fluctuation stability for given range of episode lengths
%
%  If epimin = 8 and epimax = 32, then bandit fluctuation stability ~ 3. This
%  corresponds to an average episode length of 16 trials, and should be what
%  will be used in the experiment.

% clear workspace
clear all java
close all hidden
clc

% initialize random number generator
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

% experiment parameters
epimin =  8; % minimum episode length (default: 12 trials)
epimax = 32; % maximum episode length (default: 48 trials)
nepi   = 1e4; % number of episodes to be generated (default: 1e4)

tau_vec = 0:0.2:6; % list of state stability values
ntau = length(tau_vec);

epicdf = nan(ntau,epimax+1); % episode length cdf
epiprp = nan(ntau,1); % proportion of episode lengths which meet desired range
epiavg = nan(ntau,1); % average length of episodes longer than minimum length epimin

% simulate episodes
hbar = waitbar(0,'');
for itau = 1:ntau
    tau = tau_vec(itau);
    get_pr = @(p)betarnd(1+p*exp(tau),1+(1-p)*exp(tau));
    waitbar(itau/ntau,hbar,sprintf('state stability = %.1f',tau));
    % generate episodes
    epilen = nan(1,nepi);
    pr = zeros(1,epimax+2);
    for i = 1:nepi
        pr(:) = 0;
        pr(1) = get_pr(0.5);
        if pr(1) < 0.5, pr(1) = 1-pr(1); end
        for j = 2:epimax+2
            pr(j) = get_pr(pr(j-1));
            if pr(j) < 0.5
                break
            end
        end
        epilen(i) = j-1;
    end
    % store output
    [nh,xh] = hist(epilen,0:epimax+1);
    epicdf(itau,:) = cumsum(nh(1:end-1))/nepi;
    epiprp(itau) = mean(epilen >= epimin & epilen <= epimax);
    epiavg(itau) = mean(epilen(epilen >= epimin & epilen <= epimax));
end
close(hbar);

% determine bandit fluctuation stability
epiprp_smo = smoothn(epiprp,1); % robust spline smoothing
epiavg_smo = smoothn(epiavg,1); % robust spline smoothing
tau_hat = tau_vec(1):0.001:tau_vec(end);
epiprp_hat = interp1(tau_vec,epiprp_smo,tau_hat,'spline');
tau_max = fminbnd(@(p)-interp1(tau_vec,epiprp_smo,p,'spline'),tau_vec(1),tau_vec(end));
fprintf('best state stability value = %.2f\n\n',tau_max);
% plot results
figure;
hold on
plot(tau_hat,epiprp_hat,'k-');
plot(tau_vec,epiprp,'ko');
plot(tau_max*[1,1],ylim,'r-');
hold off
xlabel('bandit fluctuation stability');
ylabel('proportion of valid episodes');
figure;
hold on
plot(tau_vec,epiavg_smo,'k-');
plot(tau_vec,epiavg,'ko');
plot(tau_max*[1,1],ylim,'r-');
hold off
xlabel('bandit fluctuation stability');
ylabel('average episode length');

%% Check impact of bandit sampling stability
%
%  This parameter allows to control task difficulty independently from the
%  temporal volatility of bandit values (controlled by the bandit fluctuation
%  stability parameter tau_fluc).
%
%  Values of 1 (hard) and 2 (easy) will be used in the experiment.

% clear workspace
clear all java
close all hidden
clc

tau_samp = 2; % bandit sampling stability
nsim = 1e5; % number of simulations (default: 1e5)

get_pr = @(p,t)betarnd(1+p*exp(t),1+(1-p)*exp(t));
pvec = (01:99)/100;

% create histogram
nh = nan(99,99); % value histogram
ph = nan(2,99); % value statistics (maximum a posteriori & mean)
for i = 1:length(pvec)
    % sample values
    psmp = get_pr(pvec(i),tau_samp(ones(nsim,1)));
    % get value histogram
    nh(:,i) = hist(psmp,pvec);
    [~,j] = max(ksdensity(psmp,pvec));
    ph(1,i) = pvec(j); % get maximum a posteriori
    ph(2,i) = mean(psmp); % get mean
end

% plot value histogram matrix
figure;
imagesc(01:99,01:99,nh);
hold on
xlim(xlim);
ylim(ylim);
plot(xlim,ylim,'w--');
hold off
set(gca,'PlotBoxAspectRatio',[1,1,1],'FontSize',14);
set(gca,'YDir','normal');
set(gca,'XTick',0:20:100,'YTick',0:20:100);
xlabel('bandit mean');
ylabel('bandit sample');

% plot value statistics
figure;
hold on
xlim([0,1]);
ylim([0,1]);
plot([0,1],[0,1],'k-');
plot(pvec,ph(1,:),'o'); % plot maximum a posteriori
plot(pvec,ph(2,:),'-'); % plot mean
hold off
set(gca,'PlotBoxAspectRatio',[1,1,1],'FontSize',14);
set(gca,'YDir','normal');
set(gca,'XTick',0:0.2:1,'YTick',0:0.2:1);
xlabel('bandit mean');
ylabel('bandit sample');

%% Generate large amount of data
%
%  Default parameters for the experiment:
%    * epimin   = 8  |
%    * epimax   = 32 | => average episode length = 16 trials
%    * tau_fluc = 3  |
%    * tau_samp = 1 (hard) or 2 (easy)
%    * anticorr = true or false
%
%  The performance of a simple RL model with knowledge of chosen and unchosen
%  rewards is computed across learning rates to identify the optimal learning
%  rate in each condition. The three parameters responsible for the generation
%  of episodes (epimin, epimax and tau_fluc) should not be changed across
%  conditions.

% clear workspace
clear all java
clc

% initialize random number generator
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

% set configuration parameters
cfg          = [];
cfg.epimin   = 8; % minimum episode length (default: 8 trials)
cfg.epimax   = 32; % maximum episode length (default: 32 trials)
cfg.tau_fluc = 3; % bandit fluctuation stability (default: 3)
cfg.tau_samp = 2; % bandit sampling stability (1 or 2)
cfg.anticorr = true; % bandit anti-correlation? (true or false)

% generate data
data = gen_data(cfg);
ntrl = length(data.ps);
fprintf('generated %d trials\n',ntrl);

% generate random samples for softmax
erand = rand(1,ntrl);

%%

% update policy for unchosen Q-value
% 0: do not update
% 1: update towards 50
% 2: update towards mean
% 3: update towards 100 minus chosen value
% 4: update with negative of chosen update
upd_unc = 3;

alpha_vec = linspace(0,1,20);
beta_vec = linspace(0,10,20);
vs_mat = nan(numel(alpha_vec),numel(beta_vec));
ds_mat = nan(numel(alpha_vec),numel(beta_vec));

for ivec = 1:numel(alpha_vec)
    ivec
    for jvec = 1:numel(beta_vec)
        % set parameter values
        alpha = alpha_vec(ivec);
        beta = beta_vec(jvec);
        % initialize variables for current simulation
        q = zeros(2,ntrl); % Q-values
        resp = zeros(1,ntrl); % responses (1 or 2)
        vs = zeros(1,ntrl); % reward value
        ds = zeros(1,ntrl); % reward value difference (chosen minus unchosen)
        % first trial
        q(:,1) = 50;
        resp(1) = 1+(erand(1) > 0.5);
        % subsequent trials
        for itrl = 2:ntrl
            if resp(itrl-1) == 1
                % compute rewards
                vs(itrl-1) = data.ps(itrl-1); % value
                ds(itrl-1) = data.ps(itrl-1)-data.qs(itrl-1); % value difference
                % update chosen Q-value
                q(1,itrl) = q(1,itrl-1)+alpha*(data.ps(itrl-1)-q(1,itrl-1));
                % update unchosen Q-value
                if upd_unc == 0 % do not update
                    q(2,itrl) = q(2,itrl-1);
                elseif upd_unc == 1 % update towards 50
                    q(2,itrl) = q(2,itrl-1)+alpha*(50-q(2,itrl-1));
                elseif upd_unc == 2 % update towards mean
                    q(2,itrl) = q(2,itrl-1)+alpha*(0.5*(q(1,itrl-1)+q(2,itrl-1))-q(2,itrl-1));
                elseif upd_unc == 3 % update towards 100 minus chosen value
                    q(2,itrl) = q(2,itrl-1)+alpha*(100-data.ps(itrl-1)-q(2,itrl-1));
                elseif upd_unc == 4 % update with negative of chosen update
                    q(2,itrl) = q(2,itrl-1)-alpha*(data.ps(itrl-1)-q(1,itrl-1));
                else
                    error('undefined update policy!');
                end
            else
                % compute rewards
                vs(itrl-1) = data.qs(itrl-1); % value
                ds(itrl-1) = data.qs(itrl-1)-data.ps(itrl-1); % value difference
                % update chosen Q-value
                q(2,itrl) = q(2,itrl-1)+alpha*(data.qs(itrl-1)-q(2,itrl-1));
                % update unchosen Q-value
                if upd_unc == 0 % do not update
                    q(1,itrl) = q(1,itrl-1);
                elseif upd_unc == 1 % update towards 50
                    q(1,itrl) = q(1,itrl-1)+alpha*(50-q(1,itrl-1));
                elseif upd_unc == 2 % update towards mean
                    q(1,itrl) = q(1,itrl-1)+alpha*(0.5*(q(1,itrl-1)+q(2,itrl-1))-q(1,itrl-1));
                elseif upd_unc == 3 % update towards 100 minus chosen value
                    q(1,itrl) = q(1,itrl-1)+alpha*(100-data.qs(itrl-1)-q(1,itrl-1));
                elseif upd_unc == 4 % update with negative of chosen update
                    q(1,itrl) = q(1,itrl-1)-alpha*(data.qs(itrl-1)-q(2,itrl-1));
                else
                    error('undefined update policy!');
                end
            end
            % respond using softmax policy
            p = 1/(1+exp(-beta*(q(1,itrl)-q(2,itrl))));
            resp(itrl) = 1+(erand(itrl) > p);
        end
        % compute rewards
        if resp(end) == 1
            vs(end) = data.ps(end);
            ds(end) = data.ps(end)-data.qs(end);
        else
            vs(end) = data.qs(end);
            ds(end) = data.qs(end)-data.ps(end);
        end
        % store mean reward
        vs_mat(ivec,jvec) = mean(vs);
        ds_mat(ivec,jvec) = mean(ds);
    end
end

figure;
imagesc(beta_vec,alpha_vec,vs_mat-50);
axis square
set(gca,'YDir','normal','TickDir','out');
set(gca,'FontSize',14);
set(gca,'CLim',[0,7]);
xlabel('beta');
ylabel('alpha');

%%
%
%  Here, a deterministic/hardmax response selection policy is used. This is a
%  priori optimal if upd_unc > 0, but clearly suboptimal if upd_unc = 0.

% update policy for unchosen Q-value
% 0: do not update
% 1: update towards global mean (50)
% 2: update towards current mean
% 3: update towards mirror of chosen value wrt global mean (50)
% 4: update towards mirror of chosen value wrt current mean
upd_unc = 4;

alpha_vec = 0:0.05:1; % learning rate for chosen Q-value
alphu_vec = 0:0.05:1; % learning rate for unchosen Q-value

% initialize reward matrices
vs_mat = nan(numel(alpha_vec),numel(alphu_vec));
ds_mat = nan(numel(alpha_vec),numel(alphu_vec));

for ivec = 1:numel(alpha_vec)
    ivec
    for jvec = 1:numel(alphu_vec)
        % set parameter values
        alpha = alpha_vec(ivec);
        alphu = alphu_vec(jvec);
        % initialize variables for current simulation
        q = zeros(2,ntrl); % Q-values
        resp = zeros(1,ntrl); % responses (1 or 2)
        vs = zeros(1,ntrl); % chosen value
        ds = zeros(1,ntrl); % relative value (chosen minus unchosen)
        % first trial
        q(:,1) = 50;
        resp(1) = 1+(erand(1) > 0.5);
        % subsequent trials
        for itrl = 2:ntrl
            if resp(itrl-1) == 1
                % compute rewards
                vs(itrl-1) = data.ps(itrl-1); % chosen value
                ds(itrl-1) = data.ps(itrl-1)-data.qs(itrl-1); % relative value
                % update chosen Q-value
                q(1,itrl) = q(1,itrl-1)+alpha*(data.ps(itrl-1)-q(1,itrl-1));
                % update unchosen Q-value
                if     upd_unc == 0 % do not update
                    q(2,itrl) = q(2,itrl-1);
                elseif upd_unc == 1 % update towards global mean (50)
                    q(2,itrl) = q(2,itrl-1)+alphu*(50-q(2,itrl-1));
                elseif upd_unc == 2 % update towards current mean
                    q(2,itrl) = q(2,itrl-1)+alphu*0.5*(q(1,itrl-1)-q(2,itrl-1));
                elseif upd_unc == 3 % update towards mirror of chosen value wrt global mean (50)
                    q(2,itrl) = q(2,itrl-1)+alphu*(100-data.ps(itrl-1)-q(2,itrl-1));
                elseif upd_unc == 4 % update towards mirror of chosen value wrt current mean
                    q(2,itrl) = q(2,itrl-1)+alphu*(q(1,itrl-1)-data.ps(itrl-1));
                else
                    error('undefined update policy!');
                end
            else
                % compute rewards
                vs(itrl-1) = data.qs(itrl-1); % chosen value
                ds(itrl-1) = data.qs(itrl-1)-data.ps(itrl-1); % relative value
                % update chosen Q-value
                q(2,itrl) = q(2,itrl-1)+alpha*(data.qs(itrl-1)-q(2,itrl-1));
                % update unchosen Q-value
                if     upd_unc == 0 % do not update
                    q(1,itrl) = q(1,itrl-1);
                elseif upd_unc == 1 % update towards global mean (50)
                    q(1,itrl) = q(1,itrl-1)+alphu*(50-q(1,itrl-1));
                elseif upd_unc == 2 % update towards current mean
                    q(1,itrl) = q(1,itrl-1)+alphu*0.5*(q(2,itrl-1)-q(1,itrl-1));
                elseif upd_unc == 3 % update towards mirror of chosen value wrt global mean (50)
                    q(1,itrl) = q(1,itrl-1)+alphu*(100-data.qs(itrl-1)-q(1,itrl-1));
                elseif upd_unc == 4 % update towards mirror of chosen value wrt current mean
                    q(1,itrl) = q(1,itrl-1)+alphu*(q(2,itrl-1)-data.qs(itrl-1));
                else
                    error('undefined update policy!');
                end
            end
            % respond using hardmax policy
            resp(itrl) = 1+(q(1,itrl) < q(2,itrl));
        end
        % compute rewards
        if resp(end) == 1
            vs(end) = data.ps(end); % chosen value
            ds(end) = data.ps(end)-data.qs(end); % relative value
        else
            vs(end) = data.qs(end); % chosen value
            ds(end) = data.qs(end)-data.ps(end); % relative value
        end
        % store mean reward
        vs_mat(ivec,jvec) = mean(vs); % chosen value
        ds_mat(ivec,jvec) = mean(ds); % relative value
    end
end

% plot simulation results
figure;
% plot chosen value wrt model parameters
subplot(1,2,1);
imagesc(alphu_vec,alpha_vec,vs_mat);
axis square
set(gca,'YDir','normal','TickDir','out');
set(gca,'FontSize',14);
set(gca,'CLim',[vs_mat(end,end),max(vs_mat(:))]);
xlabel('alphu');
ylabel('alpha');
% plot relative value wrt model parameters
subplot(1,2,2);
imagesc(alphu_vec,alpha_vec,ds_mat);
axis square
set(gca,'YDir','normal','TickDir','out');
set(gca,'FontSize',14);
set(gca,'CLim',[ds_mat(end,end),max(ds_mat(:))]);
xlabel('alphu');
ylabel('alpha');

%%

% run model across learning rates
alpha_vec = 0:0.01:1; % learning rates
vs_vec = nan(size(alpha_vec)); % absolute rewards
ds_vec = nan(size(alpha_vec)); % relative rewards
for ivec = 1:length(alpha_vec)
    alpha = alpha_vec(ivec);
    q = zeros(2,ntrl+1);
    q(:,1) = 50;
    for itrl = 1:ntrl
        q(1,itrl+1) = q(1,itrl)+alpha*(data.ps(itrl)-q(1,itrl));
        q(2,itrl+1) = q(2,itrl)+alpha*(data.qs(itrl)-q(2,itrl));
    end
    vs = zeros(1,ntrl);
    ds = zeros(1,ntrl);
    for itrl = 2:ntrl
        if q(1,itrl) > q(2,itrl)
            vs(itrl) = data.ps(itrl);
            ds(itrl) = data.ps(itrl)-data.qs(itrl);
        else
            vs(itrl) = data.qs(itrl);
            ds(itrl) = data.qs(itrl)-data.ps(itrl);
        end
    end
    vs_vec(ivec) = mean(vs);
    ds_vec(ivec) = mean(ds);
end
% plot results
pbar = 1;
figure('Color','white');
hold on
xlim([0,1]);
ylim([0,20]);
plot(alpha_vec,vs_vec-50,'r-');
plot(alpha_vec,ds_vec,'b-');
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',14);
set(gca,'XTick',0:0.2:1);
set(gca,'YTick',0:5:20);
xlabel('learning rate');
ylabel('reward');

% compute relative power
ps = data.ps;
qs = data.qs;
Fs = 1;
ns = 96;
demean = @(x)x-mean(x(:));
shuffle = @(x)x(randperm(numel(x)));
% plot results
pbar = 1;
figure('Color','white');
rgb = colormap('lines');
hold on
xlim([0,0.2]);
ylim([-4,+10]);
x = ps(:);
[pxx,f] = pwelch(demean(x),ns,[],[],Fs);
px0 = pwelch(demean(shuffle(x)),ns,[],[],Fs);
px = pow2db(pxx)-pow2db(px0);
plot(f(2:end-1),px(2:end-1),'-','Color',rgb(1,:));
x = qs(:);
[pxx,f] = pwelch(demean(x),ns,[],[],Fs);
px0 = pwelch(demean(shuffle(x)),ns,[],[],Fs);
px = pow2db(pxx)-pow2db(px0);
plot(f(2:end-1),px(2:end-1),'-','Color',rgb(2,:));
x = ps(:)-qs(:);
[pxx,f] = pwelch(demean(x),ns,[],[],Fs);
px0 = pwelch(demean(shuffle(x)),ns,[],[],Fs);
px = pow2db(pxx)-pow2db(px0);
plot(f(2:end-1),px(2:end-1),'k-');
x = ps(:)+qs(:);
[pxx,f] = pwelch(demean(x),ns,[],[],Fs);
px0 = pwelch(demean(shuffle(x)),ns,[],[],Fs);
px = pow2db(pxx)-pow2db(px0);
plot(f(2:end-1),px(2:end-1),'-','Color',[0.5,0.5,0.5]);
plot(xlim,[0,0],'k:');
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',14);
set(gca,'XTick',0:0.05:0.2);
set(gca,'YTick',-3:3:9);
xlabel('frequency');
ylabel('relative power (dB)');

%% Generate single block
%
%  Default parameters for the experiment:
%    * epimin   = 8  |
%    * epimax   = 32 | => average episode length = 16 trials
%    * tau_fluc = 3  |
%    * nepi     = 6
%    * ntrl     = 96 (6 episodes x 16 trials/episode)
%    * tau_samp = 1 (hard) or 2 (easy)
%    * anticorr = true or false
%
%  The 8 blocks of the task should be generated offline beforehand and their
%  order counter-balanced across subjects. Training blocks should be generated
%  with 3 episodes (i.e. 48 trials) and a bandit sampling stability tau_samp
%  increased to 3 (in comparison to 1 or 2 in the rest of the experiment).
%
%  The experiment has:
%    * 4 training/practice blocks, each lasting 48 trials
%    * 8 test blocks, each lasting 96 trials
%
%  In the training, anticorr is counter-balanced across subjects but we always
%  present the block with full feedbacks before the one with partial feedback,
%  e.g., anticorr/full anticorr/partial uncorr/full uncorr/partial. There are
%  thus only two types of practice across participants.
%
%  By contrast, in the test blocks, everything is counter-balanced across
%  participants. Slots machines can be *very* or *weakly* variable (tau_samp),
%  *linked* or *independent* from each other (anticorr) and feedback about the
%  unchosen slot machine can be *revealed* or *hidden*.

% clear workspace
clear all java
close all hidden
clc

% initialize random number generator
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

% set configuration parameters
cfg          = [];
cfg.epimin   = 8; % minimum episode length (default: 8 trials)
cfg.epimax   = 32; % maximum episode length (default: 32 trials)
cfg.nepi     = 6; % number of episodes to be generated (default: 6)
cfg.ntrl     = 96; % number of trials (default: 96, can be left empty)
cfg.tau_fluc = 3; % bandit fluctuation stability (default: 3)
cfg.tau_samp = 2; % bandit sampling stability (1 or 2)
cfg.anticorr = false; % bandit anti-correlation? (true or false)
cfg.pgen     = 0.005; % bandit generation precision (default: 0.005)

% generate block
blck = gen_blck(cfg);

pbar = 3; % plot box aspect ratio

% show bandit values
figure;
rgb = colormap('lines');
hold on
xlim([0,97]);
ylim([0,100]);
plot(blck.pp,'-','Color',rgb(1,:)*0.5+0.5);
plot(blck.ps,'o','MarkerFaceColor',rgb(1,:),'MarkerSize',4,'Color',rgb(1,:));
plot(blck.qq,'-','Color',rgb(2,:)*0.5+0.5);
plot(blck.qs,'o','MarkerFaceColor',rgb(2,:),'MarkerSize',4,'Color',rgb(2,:));
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',14);
set(gca,'XTick',10:10:90);
set(gca,'YTick',0:20:100);
xlabel('trial number');
ylabel('bandit value');

% show mean value
figure;
rgb = colormap('lines');
hold on
xlim([0,97]);
ylim([0,100]);
plot(0.5*(blck.pp+blck.qq),'-','Color',0.5*[1,1,1]);
plot(0.5*(blck.ps+blck.qs),'ko','MarkerFaceColor',[0,0,0],'MarkerSize',4);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',14);
set(gca,'XTick',10:10:90);
set(gca,'YTick',0:20:100);
xlabel('trial number');
ylabel('mean value');

% show value difference
figure;
hold on
xlim([0,97]);
ylim([-100,100]);
plot(xlim,[0,0],'k-');
plot(blck.pp-blck.qq,'-','Color',0.5*[1,1,1]);
plot(blck.ps-blck.qs,'ko','MarkerFaceColor',[0,0,0],'MarkerSize',4);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',14);
set(gca,'XTick',10:10:90);
set(gca,'YTick',-100:50:+100);
xlabel('trial number');
ylabel('value difference');
