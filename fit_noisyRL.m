function [out] = fit_noisyRL(cfg)
%  FIT_NOISYRL  Fit noisy RL model to two-armed bandit task data
%
%  Usage: [out] = FIT_NOISYRL(cfg)
%
%  The configuration structure cfg should contain the following fields:
%    * resp   - the provided responses (1 or 2)
%    * rew    - the reward values associated with each option (in [0,1])
%    * trl    - the trial number in the current block
%    * fbtype - the feedback type (1:partial or 2:complete)
%    * nsmp   - the number of samples used by the particle filter
%
%  The configuration structure cfg can additionally contain a cue field,
%  which indicates whether the trial corresponds to a cued trial in which
%  the subject was required to select a pre-cued option (cue = 1) rather
%  than choosing the option he/she wanted to sample (cue = 0). Observed
%  responses in these cued trials are uninformative. The function assumes
%  that the task contains no cued trials if no cue field is found in the
%  configuration structure.
%
%  The noisy RL model has four free parameters:
%    * alphac - the learning rate associated with the chosen option
%    * alphau - the learning rate associated with the unchosen option
%    * zeta   - the scaling of learning noise with the prediction error
%    * ksi    - the constant term of learning noise
%    * tau    - the temperature of the softmax action selection policy
%
%  Any combination of these parameters can be fixed to desired values, and
%  not fitted, by entering them as additional fields in the configuration
%  structure cfg. When fitted, the prior functions for each parameter can be
%  found and modified in the code below.
%
%  The RL model can be made to learn the policy (the difference in value
%  between option 1 and option 2) rather than the values associated with each
%  option by setting alphau to not-a-number in the configuration structure.
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
%  1/ noisy RL model
%  Findling, C., Skvortsova, V., Dromnelle R., Palminteri S., and Wyart, V.
%  (2018). "Computational noise in reward-guided learning drives behavioral
%  variability in volatile environments." bioRxiv, doi:10.1101/439885.
%
%  2/ VBMC toolbox
%  Acerbi, L. (2018). "Variational Bayesian Monte Carlo". Advances in Neural
%  Information Processing Systems #31 (NeurIPS 2018), pp. 8213-8223.
%
%  Valentin Wyart <valentin.wyart@ens.fr>

% check required parameters
if ~all(isfield(cfg,{'resp','rew','trl','fbtype','nsmp'}))
    error('Incomplete configuration structure!');
end
if ~ismember(cfg.fbtype,[1,2])
    error('Invalid feedback type (should be 1 or 2)!');
end
if ~all(ismember(cfg.resp,[1,2]))
    error('Invalid responses (should be 1 or 2)!');
end
if ~all(cfg.rew(:) >= 0 & cfg.rew(:) <= 1)
    error('Invalid reward values (should be in [0,1] range)!');
end
% check optional parameters
if ~isfield(cfg,'cue')
    cfg.cue = zeros(size(cfg.resp));
end
if ~isfield(cfg,'verbose')
    cfg.verbose = false;
end

% get data
resp = cfg.resp; % response (1 or 2)
rew  = cfg.rew; % rewards
trl  = cfg.trl; % trial number in current block
cue  = cfg.cue; % cued trial?

ntrl = numel(trl); % number of trials
nsmp = cfg.nsmp; % number of samples used by particle filter

% account for feedback type (1:partial 2:complete)
fbtype = cfg.fbtype;
if fbtype == 1
    % set unchosen rewards to overall mean
    indx = sub2ind([ntrl,2],(1:ntrl)',3-resp);
    rew(indx) = 0.5;
end

% check whether policy learning
learn_policy = false;
if isfield(cfg,'alphau') && isnan(cfg.alphau)
    learn_policy = true;
end

% define model parameters
pnam = {}; % name
pmin = []; % minimum value
pmax = []; % maximum value
pfun = {}; % log-prior function
pini = []; % initial value
pplb = []; % plausible lower bound
ppub = []; % plausible upper bound
% 1/ learning rate of chosen option
pnam{1,1} = 'alphac';
pmin(1,1) = 0;
pmax(1,1) = 1;
pfun{1,1} = @(x)betapdf(x,1,1);
pini(1,1) = betastat(1,1);
pplb(1,1) = betainv(0.15,1,1);
ppub(1,1) = betainv(0.85,1,1);
% 2/ learning rate of unchosen option
pnam{1,2} = 'alphau';
pmin(1,2) = 0;
pmax(1,2) = 1;
pfun{1,2} = @(x)betapdf(x,1,1);
pini(1,2) = betastat(1,1);
pplb(1,2) = betainv(0.15,1,1);
ppub(1,2) = betainv(0.85,1,1);
% 3/ learning noise - scaling w/ prediction error
pnam{1,3} = 'zeta';
pmin(1,3) = 0.001;
pmax(1,3) = 10;
pfun{1,3} = @(x)gampdf(x,4,0.125);
pini(1,3) = gamstat(4,0.125);
pplb(1,3) = gaminv(0.15,4,0.125);
ppub(1,3) = gaminv(0.85,4,0.125);
% 4/ learning noise - constant term
pnam{1,4} = 'ksi';
pmin(1,4) = 0.001;
pmax(1,4) = 10;
pfun{1,4} = @(x)gampdf(x,4,0.025);
pini(1,4) = gamstat(4,0.025);
pplb(1,4) = gaminv(0.15,4,0.025);
ppub(1,4) = gaminv(0.85,4,0.025);
% 5/ softmax temperature
pnam{1,5} = 'tau';
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

% define free parameters
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
nruns = 1;
vp = []; elbo = []; elbo_sd = []; exitflag = [];
for irun = 1:nruns
    [vp{irun},elbo(irun),elbo_sd(irun),exitflag(irun),output] = vbmc(@(x)fun(x),pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,options);
end
%[vp,elbo,~,exitflag,output] = vbmc(@(x)fun(x),pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,options);

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

% set unchosen learning rate if policy learning
if learn_policy
    out.alphau = out.alphac;
end

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
[~,out.pt,out.et,out.qt,out.zt] = getll(out.alphac,out.alphau,out.zeta,out.ksi,out.tau);

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

    function [ll,p,e,q,z] = getll(alphac,alphau,zeta,ksi,tau)
        % run particle filter
        p = zeros(ntrl,nsmp); % response probability
        e = zeros(ntrl,2,nsmp); % prediction errors
        q = zeros(ntrl,2,nsmp); % filtered Q-values
        z = zeros(ntrl,2,nsmp); % filtered learning errors
        for itrl = 1:ntrl
            % 1st response of each block (uninformative)
            if trl(itrl) == 1
                p(itrl,:) = 0.5;
                q(itrl,:,:) = 0.5;
                continue
            end
            % 1/ update Q-values
            q(itrl,:,:) = q(itrl-1,:,:);
            if learn_policy
                alpha = [alphac,alphac];
            else
                if resp(itrl-1) == 1
                    alpha = [alphac,alphau];
                else
                    alpha = [alphau,alphac];
                end
            end
            for iopt = 1:2
                e(itrl,iopt,:) = rew(itrl-1,iopt)-q(itrl,iopt,:); % prediction error
                s(iopt,:) = sqrt(zeta^2*e(itrl,iopt,:).^2+ksi^2); % learning noise s.d.
                q(itrl,iopt,:) = q(itrl,iopt,:)+alpha(iopt)*e(itrl,iopt,:); % exact learning
            end
            
            q0 = q(itrl,:,:);
            if cue(itrl) == 0 % free trial
                % 2/ compute response probability
                qd = reshape(q(itrl,1,:)-q(itrl,2,:),[1,nsmp]);
                supd = sqrt(sum(s.^2,1));
                ssel = pi/sqrt(6)*tau;
                sd = sqrt(supd.^2+ssel^2);
                p(itrl,:) = 1-normcdf(0,qd,sd);
                % 3/ resample Q-values wrt observed response
                if zeta > 0 || ksi > 0
                    qt = resample(squeeze(q(itrl,:,:)),s,resp(itrl));
                    q(itrl,:,:) = normrnd( ...
                        (ssel^2*squeeze(q(itrl,:,:))+supd.^2.*qt)./(supd.^2+ssel^2), ...
                        sqrt(supd.^2*ssel^2/(supd.^2+ssel^2)));
                end
            else % cued trial (uninformative)
                p(itrl,:) = 0.5;
                q(itrl,:,:) = normrnd(squeeze(q(itrl,:,:)),s);
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
