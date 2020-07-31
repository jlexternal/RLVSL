function [out] = fit_RLpSL2_rlvsl(cfg)
%  FIT_RLPSL2_RLVSL  Fit noisy RL model with prior bias to two-armed structured
%                    bandit task data
%
%  Usage: [out] = FIT_RLPSL2_RLVSL(cfg)
%
%  The configuration structure cfg should contain the following fields:
%    * resp   - the provided responses (1 or 2)
%    * rew    - the reward values associated with each option (in [0,1])
%    * trl    - the trial number in the current block
%    * fbtype - the feedback type (1:partial or 2:complete)
%    * nsmp   - the number of samples used by the particle filter
%
%
%  The noisy RL model has four free parameters:
%    * zeta   - the scaling of learning noise with the prediction error
%    * alpha  - the asymptote of the learning rate of the Kalman Filter
%    * prior - the structure learned prior bias at each decision
%
%   Disabled parameters (kept for potential future use)
%    * tau    - the temperature of the softmax action selection policy
%    * ksi    - the constant term of learning noise
%
%  Any combination of these parameters can be fixed to desired values, and
%  not fitted, by entering them as additional fields in the configuration
%  structure cfg. When fitted, the prior functions for each parameter can be
%  found and modified in the code below.
%
%  The function returns an output structure out which contains the values of
%  each fitted parameter corresponding to the posterior mode, the ELBO model
%  evidence metric (expected lower bound on log-marginal likelihood), and
%  additional metrics about posterior distributions of parameter values.
%
%  The function requires the VBMC toolbox (https://github.com/lacerbi/vbmc)
%  to be installed (i.e., added to MATLAB path).
%
%  References:
%
%  1/ basic noisy RL model
%  Findling, C., Skvortsova, V., Dromnelle R., Palminteri S., and Wyart, V.
%  (2018). "Computational noise in reward-guided learning drives behavioral
%  variability in volatile environments." bioRxiv, doi:10.1101/439885.
%
%  2/ VBMC toolbox
%  Acerbi, L. (2018). "Variational Bayesian Monte Carlo". Advances in Neural
%  Information Processing Systems #31 (NeurIPS 2018), pp. 8213-8223.
%
%  Jun Seok Lee <jlexternal@gmail.com>, Valentin Wyart <valentin.wyart@ens.fr>

% check required parameters
if ~all(isfield(cfg,{'resp','rew','trl','nsmp'}))
    error('Incomplete configuration structure!');
end
if ~all(ismember(cfg.resp,[1,2]))
    error('Invalid responses (should be 1 or 2)!');
end
if ~all(cfg.rew(:) >= 0 & cfg.rew(:) <= 1)
    error('Invalid reward values (should be in [0,1] range)!');
end
% check optional parameters
if ~isfield(cfg,'verbose')
    cfg.verbose = false;
end

% get data
resp = cfg.resp; % response (1-correct or 2-incorrect)
rew  = cfg.rew;  % rewards
trl  = cfg.trl;  % trial number in current block
stgt = cfg.stgt; % s.d. of target distribution

ntrl = numel(trl);  % number of trials
nsmp = cfg.nsmp;    % number of samples used by particle filter

% set unchosen rewards to overall mean
indx = sub2ind([ntrl,2],(1:ntrl)',3-resp);
rew(indx) = 0.5;

% define model parameters
pnam = {}; % name
pmin = []; % minimum value
pmax = []; % maximum value
pfun = {}; % log-prior function
pini = []; % initial value
pplb = []; % plausible lower bound
ppub = []; % plausible upper bound

% 1/ learning noise - scaling w/ prediction error
pnam{1,1} = 'zeta';
pmin(1,1) = 0.001;
pmax(1,1) = 10;
%pfun{1,1} = @(x)gampdf(x,4,0.125); 
pfun{1,1} = @(x)gampdf(x,2.5,0.125); % JL - testing different prior distrib.
pini(1,1) = gamstat(4,0.125);
pplb(1,1) = gaminv(0.15,4,0.125);
ppub(1,1) = gaminv(0.85,4,0.125);
% 2/ learning rate asympote
pnam{1,2} = 'alpha';
pmin(1,2) = 0;
pmax(1,2) = 1;
pfun{1,2} = @(x)betapdf(x,1,1);
pini(1,2) = betastat(1,1);
pplb(1,2) = betainv(0.15,1,1);
ppub(1,2) = betainv(0.85,1,1);
% 3/ structure learned prior (in units of probability)
pnam{1,3} = 'prior';
pmin(1,3) = 0;
pmax(1,3) = 1;
pfun{1,3} = @(x)betapdf(x,1,1);
pini(1,3) = betastat(1,1);
pplb(1,3) = betainv(0.15,1,1);
ppub(1,3) = betainv(0.85,1,1);
% 4/ softmax temperature
pnam{1,4} = 'tau';
pmin(1,4) = 0.001;
pmax(1,4) = 10;
pfun{1,4} = @(x)gampdf(x,4,0.025);
pini(1,4) = gamstat(4,0.025);
pplb(1,4) = gaminv(0.15,4,0.025);
ppub(1,4) = gaminv(0.85,4,0.025);
% 5/ learning noise - constant term
pnam{1,5} = 'ksi';
pmin(1,5) = 0.001;
pmax(1,5) = 10;
pfun{1,5} = @(x)gampdf(x,4,0.025);
pini(1,5) = gamstat(4,0.025);
pplb(1,5) = gaminv(0.15,4,0.025);
ppub(1,5) = gaminv(0.85,4,0.025);

% define fixed parameters
npar = numel(pnam);
pfix = cell(1,npar);
for i = 1:npar
    if isfield(cfg,pnam{i})
        pfix{i} = cfg.(pnam{i});
    end
end
if nnz(cellfun(@isempty,pfix)) == 0
    error('All parameters fixed, nothing to fit!');
end

% define free parameters (to be input to vbmc)
ifit = cell(1,npar);
pfit_ini = [];
pfit_min = [];
pfit_max = [];
pfit_plb = [];
pfit_pub = [];
n = 1;
for i = 1:npar
    if isempty(pfix{i}) % free parameter
        ifit{i} = n;
        pfit_ini = cat(2,pfit_ini,pini(i));
        pfit_min = cat(2,pfit_min,pmin(i));
        pfit_max = cat(2,pfit_max,pmax(i));
        pfit_plb = cat(2,pfit_plb,pplb(i));
        pfit_pub = cat(2,pfit_pub,ppub(i));
        n = n+1;
    end
end
nfit = length(pfit_ini);

% configure VBMC
options = vbmc('defaults');
if cfg.verbose, opt_disp = 'iter'; else, opt_disp = 'notify'; end
options.Display = opt_disp; % level of display
options.MaxIter = 300; % maximum number of iterations
options.MaxFunEvals = 500; % maximum number of function evaluations
options.UncertaintyHandling = 1; % noisy log-posterior function

% fit model using VBMC
nruns = 1; % set to 3 or greater if testing for convergence
vp = []; elbo = []; elbo_sd = []; exitflag = [];
for irun = 1:nruns
    [vp{irun},elbo(irun),elbo_sd(irun),exitflag(irun),output] = vbmc(@(x)fun(x),pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,options);
end

% generate 10^6 samples from the variational posterior
xsmp = vbmc_rnd(vp{1},1e6);

xmap = vbmc_mode(vp{1}); % posterior mode
xavg = mean(xsmp,1); % posterior means
xstd = std(xsmp,[],1); % posterior s.d.
xcov = cov(xsmp); % posterior covariance matrix
xmed = median(xsmp,1); % posterior medians
xiqr = quantile(xsmp,[0.25,0.75],1); % posterior interquartile ranges

% create output structure with parameter values
[~,phat] = fun(xmap); % use posterior mode
out = cell2struct(phat(:),pnam(:));

% store number of samples used by particle filter
out.nsmp = nsmp;

% store VBMC output
out.vp   = vp;
out.elbo = elbo; % ELBO (expected lower bound on log-marginal likelihood)
out.xnam = pnam(cellfun(@isempty,pfix)); % fitted parameters
out.xmap = xmap; % posterior mode
out.xavg = xavg; % posterior means
out.xstd = xstd; % posterior s.d.
out.xcov = xcov; % posterior covariance matrix
out.xmed = xmed; % posterior medians
out.xiqr = xiqr; % posterior interquartile ranges

% store additional VBMC output
out.exitflag = exitflag;
out.output   = output;

% store trajectories
[~,out.pt,out.et,out.qt,out.zt,out.kt,out.vt] = getll(out.zeta,out.alpha,out.prior,out.tau,out.ksi);

% model parameter organizing function 
function [l,pval] = fun(p)
    pval = cell(1,npar); % parameter values
    lp = 0; % log-prior
    for i = 1:npar
        if isempty(pfix{i}) % free parameter
            pval{i} = p(ifit{i});
            lp = lp+log(pfun{i}(pval{i}));
        else % fixed parameter
            pval{i} = pfix{i};
        end
    end
    ll = getll(pval{:}); % log-likelihood
    l = ll+lp; % unnormalized log-posterior
end

% calculate log-likelihood via particle filter
function [ll,p,e,q,z,k,v] = getll(zeta,alpha,prior,tau,ksi)
    % run particle filter
    p = zeros(ntrl,nsmp); % response probability
    e = zeros(ntrl,2,nsmp); % prediction errors
    q = zeros(ntrl,2,nsmp); % filtered Q-values
    z = zeros(ntrl,2,nsmp); % filtered learning errors
    qlt = zeros(ntrl,nsmp); % logit on Q-values
    slt = zeros(ntrl,nsmp); % logit on structure learned prior
    
    k = zeros(ntrl,2,nsmp); % kalman gains
    v = zeros(ntrl,2,nsmp); % covariance noise
    
    vs = stgt^2;              % sampling variance
    vn = (alpha/(1-alpha))^2; % process noise (calculated from value of asympote parameter)
    
    for itrl = 1:ntrl
        % 1st response of each block (uninformative)
        if trl(itrl) == 1
            q(itrl,:,:) = 0.5;   % KF posterior mean
            qlt(itrl,:) = 1e-6;  % 'Q logit'
            slt(itrl,:) = log(prior)-log(1-prior); % 'S logit'
            k(itrl,:,:) = 0;     % Kalman gain
            v(itrl,:,:) = 1e6;   % Covariance noise
            p(itrl,:)   = (1+exp(-(qlt(itrl,:)+slt(itrl,:)))).^-1;
            continue
        end
        
        % 1/ update Q-values
        q(itrl,:,:) = q(itrl-1,:,:);
        v(itrl,:,:) = v(itrl-1,:,:);
        
        for iopt = 1:2 % tracking for the two choices
            e(itrl,iopt,:)  = rew(itrl-1,iopt)-q(itrl,iopt,:);               % prediction error
            s(iopt,:)       = sqrt(zeta^2*e(itrl,iopt,:).^2+ksi^2);          % learning noise s.d.
            k(itrl,iopt,:)  = v(itrl,iopt,:)./(v(itrl,iopt,:)+vs);           % kalman gain update
            q(itrl,iopt,:)  = q(itrl,iopt,:)+k(itrl,iopt,:).*e(itrl,iopt,:); % exact learning
            v(itrl,iopt,:)  = (1-k(itrl,iopt,:)).*v(itrl,iopt,:)+vn;         % covariance noise update
        end
        
        %%%
        pq = 1-normcdf(0,q(ib,it,:),sqrt(vt(ib,it,:))); % probability of correct response based on Q-value
        qlt(ib,it,:) = log(pq)-log(1-pq);       % 'Q logit'
        slt(ib,it,:) = log(prior)-log(1-prior); % 'S logit'
        
        p(ib,it,:)   = (1+exp(-(qlt(ib,it,:)+slt(ib,it,:)))).^-1; % probability of correct response accounting for structure
        %%%
        
        q0 = q(itrl,:,:); % hold exact learned q value
        
        % 2/ compute response probability
        qd          = reshape(q(itrl,1,:)-q(itrl,2,:),[1,nsmp]); % difference of q-values
        supd        = sqrt(sum(s.^2,1));                         % learning noise s.d. 
        ssel        = pi/sqrt(6)*tau;                            % softmax choice s.d. (argmax'ed if tau = 0)
        sd          = sqrt(supd.^2+ssel^2);                      % s.d. of q-val difference distrib.
        
        p(itrl,:)   = 1-normcdf(0,qd,sd);                        % probability of response from strictly RL
        qlt(itrl,:) = log(p(itrl,:))-log(1-p(itrl,:)); %         % logit contribution of RL
        slt(itrl,:) = log(prior)-log(1-prior);         %         % logit contribution of SL 
        
        p(itrl,:)   = (1+exp(-(qlt(itrl,:)+slt(itrl,:)))).^-1; % % integrated probability of response

        % 3/ resample Q-values wrt observed response
        if zeta > 0 || ksi > 0
            qt = resample(squeeze(q(itrl,:,:)),s,resp(itrl));
            q(itrl,:,:) = normrnd( ...
                (ssel^2*squeeze(q(itrl,:,:))+supd.^2.*qt)./(supd.^2+ssel^2), ...
                sqrt(supd.^2*ssel^2/(supd.^2+ssel^2)));
        end
        
        % compute learning errors
        z(itrl,:,:) = q(itrl,:,:)-q0;
    end
    % average across samples
    p = mean(p,2);
    e = mean(abs(e),3); % average magnitude of prediction errors
    q = mean(q,3);
    z = mean(abs(z),3); % average magnitude of learning errors
    % compute log-likelihood
    ep = 1e-6; % epsilon
    ll = sum(log((p.*(resp == 1)+(1-p).*(resp ~= 1))*(1-ep)+0.5*ep));
end

end

function [qt] = resample(q,s,r)
    % 1/ resample Q1-Q2 from truncated normal distribution
    qd = tnormrnd(q(1,:)-q(2,:),sqrt(sum(s.^2,1)),r);
    % 2/ resample Q1 from Q1-Q2
    ax = s(1,:).^2./sum(s.^2,1);
    mx = q(1,:)-ax.*(q(1,:)-q(2,:));
    sx = sqrt(s(1,:).^2-ax.^2.*sum(s.^2,1));
    q1 = ax.*qd+normrnd(mx,sx);
    % 3/ return Q1 and Q2 = Q1-(Q1-Q2)
    qt = cat(1,q1,q1-qd);
end

function [x] = tnormrnd(m,s,d)
    % sample from truncated normal distribution
    if d == 1
        x = +rpnormv(+m,s);
    else
        x = -rpnormv(-m,s);
    end
end