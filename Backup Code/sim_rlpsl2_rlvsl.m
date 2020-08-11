% sim_rlpsl2_rlvsl

% simulate model RLPSL2 

% Experimental parameters
nb = 16;
nt = 16;

% Generative parameters of winning distribution
% with false negative rate of 25%
r_mu = 5;
r_sd = 7.413;

% Model parameters
ns      = 1;  % Number of simulated agents to generate per given parameter
zeta    = .2; % Learning noise for value learning in KF
alpha   = 0;  % Kalman gain asymptote parameter (multiplicative factor on the sampling variance) 
prior   = .5; % value for the prior probability of choosing shape A via structure learning for each condition

rew = round(normrnd(r_mu,r_sd,[nb nt ns])); % rewards

% Kalman Filter variables
q  = nan(nb,nt,ns);    % posterior mean
qlt = nan(size(q));
slt = nan(size(q));
p = nan(size(q));
c = nan(size(q));
vt = nan(nb,nt,ns);    % posterior variance
kt = zeros(nb,nt,ns);  % kalman gain
vs = r_sd^2;           % sampling variance
vd = (alpha/(1-alpha))^2;   % (perceived) process noise as multiplicative factor of vs

rpe = nan(size(q));    % store output: RPEs
    
for ib = 1:nb
    
    for it = 1:nt
        % 1st response of each block (uninformative)
        if it == 1
            q(ib,it,:)   = 0;                       % KF posterior mean
            qlt(ib,it,:) = 1e-6;                    % 'Q logit'
            slt(ib,it,:) = log(prior)-log(1-prior); % 'S logit'
            kt(ib,it,:)  = 0;                       % Kalman gain
            vt(ib,it,:)  = 1e6;                     % Covariance noise
            p(ib,it,:)   = (1+exp(-(qlt(ib,it,:)+slt(ib,it,:)))).^-1;
            for is = 1:ns
                c(ib,it,is)   = datasample([1 2],1,'Weights',[p(ib,it,is) 1-p(ib,it,is)]);
            end
            continue
        end
        
        % Update Q-value
        vt(ib,it,:) = vt(ib,it-1,:);
        q(ib,it,:)  = q(ib,it-1,:);
        
        kt(ib,it,:)  = vt(ib,it,:)./(vt(ib,it,:)+vs);     % kalman gain update
        rpe(ib,it,:) = rew(ib,it-1,:)-q(ib,it,:);         % RPE calculation
        q(ib,it,:)   = q(ib,it,:) + kt(ib,it,:).*rpe(ib,it,:).*(1+randn(1,1,ns)*zeta); % update Q-value
        vt(ib,it,:)  = (1-kt(ib,it,:)).*vt(ib,it-1,:)+vd; % covariance noise update
        
        pq = 1-normcdf(0,q(ib,it,:),sqrt(vt(ib,it,:))); % probability of correct response based on Q-value
        qlt(ib,it,:) = log(pq)-log(1-pq);       % 'Q logit'
        slt(ib,it,:) = log(prior)-log(1-prior); % 'S logit'
        
        p(ib,it,:)   = (1+exp(-(qlt(ib,it,:)+slt(ib,it,:)))).^-1; % probability of correct response accounting for structure
        for is = 1:ns
            c(ib,it,is)   = datasample([1 2],1,'Weights',[p(ib,it,is) 1-p(ib,it,is)]);
        end
        
        
    end
end

%% Plot learning curves
figure;
c4plot = c;
c4plot(c4plot==2) = 0;
if ns > 1
    shadedErrorBar(1:16,mean(mean(c4plot,1),3),std(mean(c4plot,1),1,3)/sqrt(ns),'lineprops',{'LineWidth',2},'patchSaturation',0.075);
else
    shadedErrorBar(1:16,mean(c4plot,1),std(c4plot,1,1)/sqrt(ns),'lineprops',{'LineWidth',2},'patchSaturation',0.075);
end
hold on;
yline(.5,':');
yline(1);
ylim([.3 1]);
title(sprintf('Results of model simulation (%d)\n Parameters (prior: %.2f, zeta: %.2f, asymptote: %.2f)',ns,prior,zeta,alpha));

%% Recover parameters
addpath('./Toolboxes');
% add VBMC toolbox to path
addpath('./vbmc');

out_fit = struct;

blk = kron(1:nb,ones(1,nt))';
trl = repmat((1:nt)',[nb,1]);

resp = permute(c,[2 1]);
resp = resp(:); % vectorize response matrix
resp(resp==0) = 2;

rews = nan(size(resp,1),2); % rewards of chosen & unchosen options
for i = 1:size(resp,1) % pointers on blk and trl
    rews(i,resp(i))   = (rew(blk(i),trl(i))+50);        % chosen
    rews(i,3-resp(i)) = 100-rews(i,resp(i));    % unchosen
end

% instantiate configuration structure
cfg         = [];
cfg.nsmp    = 1e3;   % number of samples
cfg.verbose = true;  % verbose VBMC output
cfg.stgt    = r_sd; %sqrt(.01^2*r_sd^2);

% fixed parameters
cfg.tau     = 1e-6; % assume argmax choice policy
cfg.ksi     = 1e-6; % assume pure Weber noise (no constant term)

% to be fitted
    % zeta  - scaling of learning noise with prediction error
    % alpha - KF learning rate asymptote
    % prior - strength of the prior belief on the correct option

cfg.resp    = resp;
cfg.rew     = rews;
cfg.trl     = trl;

out_fit = fit_RLpSL2_rlvsl(cfg);

rz = out_fit.zeta;
ra = out_fit.alpha;
rp = out_fit.prior;
fprintf('Found params: zeta: %.2f, alpha: %.2f, prior: %.2f\n',rz,ra,rp);

%% some testing

gaussian = @(x) (1/sqrt((2*pi))*exp(-x.^2/2));
skewedgaussian = @(x,alpha) 2*gaussian(x).*normcdf(alpha*x);
x = [-3:.1:3];
plot(x, gaussian(x))
hold on
plot(x, skewedgaussian(x, 4))
plot(x, skewedgaussian(x, -4))
plot(x, skewedgaussian(x, 1))
plot(x, skewedgaussian(x, -1))


