% sim_rlpsl3_rlvsl

% simulate model RLPSL3

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

%%
bpars = fmincon(@(x)abs(prior-betacdf(.5,x(1),x(2),'upper')),[1;1],[],[],[],[],1,100);

rew = round(normrnd(r_mu,r_sd,[nb nt ns]))/100; % rewards

% Kalman Filter variables
q  = nan(nb,nt,ns);    % posterior mean
qlt = nan(size(q));
slt = nan(size(q));
p = nan(size(q));
c1 = nan(size(q));
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
                c1(ib,it,is)   = datasample([1 2],1,'Weights',[p(ib,it,is) 1-p(ib,it,is)]);
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
            c1(ib,it,is)   = datasample([1 2],1,'Weights',[p(ib,it,is) 1-p(ib,it,is)]);
        end
        
        
    end
end

%% Plot learning curves
figure;
c4plot = c1;
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

%% Testing convolution

zeta    = .2; % Learning noise for value learning in KF
alpha   = .05;  % Kalman gain asymptote parameter (multiplicative factor on the sampling variance) 
prior   = .8; % value for the prior probability of choosing shape A via structure learning for each condition

t = [0:.005:1]; % support of the distributions

% Candidate structure distributions
% Beta distribution
    bpars = fmincon(@(x) abs(prior-betacdf(.5,x(1),x(2),'upper')),[1;1],[],[],[],[],1,10000); 

% Skew normal distribution 
    % Parameters to center distribution and contain 99.7% of density within [0,1]
    xi      = .5;   % location
    omega   = 1/6;  % scale
    gaussian = @(x) (1/sqrt((2*pi))*exp(-x.^2/2));
    skewgaussian = @(x,sk_alpha) (2/omega)*gaussian((x-xi)/omega).*normcdf(sk_alpha*(x-xi)/omega); % 2*pdf(x)*cdf(alpha*x)
    sk_alpha = fmincon(@(sk_alpha) abs(prior-upper_skewgaussian(t,xi,omega,sk_alpha)),1,[],[],[],[],0,[]);
   
    
% Note: Convolution of a flat prior with a peaked distribution results is also very flat. 
%       A skewed distribution may be necessary to remediate this.

sdist1 = betapdf(t,bpars(1),bpars(2));  % beta distribution as structure
sdist2 = skewgaussian(t,sk_alpha);      % skew gaussian as structure

rew = round(normrnd(r_mu,r_sd,[nt 1]))/100+.5; % rewards
vs = r_sd^2*.01^2;           % sampling variance
vd = (alpha/(1-alpha))^2;   % (perceived) process noise as multiplicative factor of vs

for it = 1:nt
    % 1st response of each block (uninformative)
    if it == 1
        q(it)   = 0.5;                     % KF posterior mean
        kt(it)  = 0;                       % Kalman gain
        vt(it)  = 1e3;                     % Covariance noise
        
        qdist = normpdf(t,q(it),vt(it));
        c1 = conv(sdist1,qdist,'same');
        c2 = conv(sdist2,qdist,'same');
        
        figure(1);
        plot(t,sdist1/sum(sdist1));
        hold on;
        xline(.5);
        plot(t,qdist/sum(qdist));
        plot(t,c1/sum(c1),'LineWidth',2);
        hold off;
        
        figure(2);
        plot(t,sdist2/sum(sdist2));
        hold on;
        xline(.5);
        plot(t,qdist/sum(qdist));
        plot(t,c2/sum(c2),'LineWidth',2);
        hold off;
        pause;
        continue;
    end

    % Update Q-value
    vt(it) = vt(it-1);
    q(it)  = q(it-1);

    kt(it)  = vt(it)./(vt(it)+vs);     % kalman gain update
    rpe(it) = rew(it-1)-q(it);         % RPE calculation
    q(it)   = q(it) + kt(it).*rpe(it).*(1+randn(1,1,ns)*zeta); % update Q-value
    vt(it)  = (1-kt(it)).*vt(it-1)+vd; % covariance noise update

    pq = 1-normcdf(0,q(it),sqrt(vt(it))); % probability of correct response based on Q-value
    
    qdist = normpdf(t,q(it),vt(it));
    c1 = conv(sdist1,qdist,'same');
    c2 = conv(sdist2,qdist,'same');
    
    figure(1);
    clf;
    plot(t,sdist1/sum(sdist1));
    hold on;
    xline(.5);
    plot(t,qdist/sum(qdist));
    plot(t,c1/sum(c1),'LineWidth',2);
    hold off; 
    
    figure(2);
    clf;
    plot(t,sdist2/sum(sdist2));
    hold on;
    xline(.5);
    plot(t,qdist/sum(qdist));
    plot(t,c2/sum(c2),'LineWidth',2);
    hold off;
    pause();
    
    end


function upper_cdf = upper_skewgaussian(x,xi,omega,sk_alpha)
    gaussian  = @(x) (1/sqrt((2*pi))*exp(-x.^2/2));
    pdf       = (2/omega)*gaussian((x-xi)/omega).*normcdf(sk_alpha*(x-xi)/omega); % 2*pdf(x)*cdf(alpha*x)
    upper_cdf = sum(pdf(round(length(x)/2):end)/sum(pdf));
end