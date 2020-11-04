function [out] = fit_noisyKF_empPrior(cfg)

% check configuration structure
if ~isfield(cfg,'rt')
    error('Missing rewards!');
end
if ~isfield(cfg,'resp')
    error('Missing responses!');
end
if ~all(isfield(cfg,{'ms','vs'}))
    error('Missing sampling statistics!');
end
if ~isfield(cfg,'empprior')
    error('Missing empirical prior (normal) distribution parameters!');
end
if ~isfield(cfg,'nscheme')
    cfg.nscheme = 'upd'; % rpe:reward prediction errors or upd:action value updates
    warning('Assuming learning noise scaling with action value updates.');
end
if ~isfield(cfg,'lscheme')
    cfg.lscheme = 'ind'; % ind:independent or sym:symmetric
    warning('Assuming independence between action values.');
end
if ~isfield(cfg,'cscheme')
    cfg.cscheme = 'qvs'; % qvs:Q-value sampling or ths:Thompson sampling
    warning('Assuming Q-value sampling.');
end
if ~isfield(cfg,'sbias_ini')
    cfg.sbias_ini = false;
    warning('Assuming no structure bias on initial action values.');
end
if ~isfield(cfg,'sbias_cor')
    cfg.sbias_cor = false;
    warning('Assuming subjective structure bias (toward 1st response).');
end
if ~isfield(cfg,'nsmp')
    cfg.nsmp = 1e3;
end
if ~isfield(cfg,'nres')
    cfg.nres = 1e3;
end
if ~isfield(cfg,'verbose')
    cfg.verbose = false;
end

% check reward scaling
if any(cfg.rt(:)) > 1
    error('Rewards should be scaled between 0 and 1!');
end
% check responses
if ~all(ismember(cfg.resp,[1,2]))
    error('Responses should be 1 or 2!');
end
% check learning noise scaling scheme
if ~ismember(cfg.nscheme,{'rpe','upd'})
    error('Undefined learning noise scaling scheme!');
end
% check counterfactual learning scheme
if ~ismember(cfg.lscheme,{'ind','sym'})
    error('Undefined counterfactual learning scheme!');
end
% check choise sampling scheme
if ~ismember(cfg.cscheme,{'qvs','ths'})
    error('Undefined choice sampling scheme!');
end

% get experiment data
resp = cfg.resp; % responses
rt   = cfg.rt; % rewards

nb = size(rt,1); % number of blocks
nt = size(rt,2); % number of trials per block
ntrl = nb*nt; % total number of trials
nsmp = cfg.nsmp; % number of samples
nres = cfg.nres; % number of re-samples for bootstrapping
minl = 0.5/ntrl; % minimum value for filtered likelihoods
verbose = cfg.verbose; % display level

% reshape experiment data
resp = reshape(resp',[],1);
rt   = reshape(rt',[],1);
trl  = reshape(repmat(1:nt,[nb,1])',[],1);

% get sampling statistics
ms = cfg.ms; % sampling mean
vs = cfg.vs; % sampling variance

% get model type
nscheme = cfg.nscheme; % noise scaling scheme ('rpe' or 'upd')
lscheme = cfg.lscheme; % latent learning scheme ('ind' or 'sym')
cscheme = cfg.cscheme; % choice sampling scheme ('qvs' or 'ths')
sbias_ini = cfg.sbias_ini; % structure bias on initial action values?
sbias_cor = cfg.sbias_cor; % structure bias toward correct action?

% empirical prior distributions
ep = cfg.empprior; 

% reparameterization functions
fk = @(v)1./(1+exp(+0.4486-log2(v/vs)*0.6282)).^0.5057;
fv = @(k)fzero(@(v)fk(v)-min(max(k,0.001),0.999),vs.*2.^[-30,+30]);

% define model parameters
pnam = {}; % name
pmin = []; % minimum value
pmax = []; % maximum value
pfun = {}; % log-prior function
pini = []; % initial value
pplb = []; % plausible lower bound
ppub = []; % plausible upper bound
% 1/ initial Kalman gain
pnam{1,1} = 'kini';
pmin(1,1) = norminv(0.01,ep(1,1),ep(1,2));
pmax(1,1) = norminv(0.99,ep(1,1),ep(1,2));
pfun{1,1} = @(x)normpdf(x,ep(1,1),ep(1,2));
pini(1,1) = normstat(ep(1,1),ep(1,2));
pplb(1,1) = norminv(0.1587,ep(1,1),ep(1,2));
ppub(1,1) = norminv(0.8413,ep(1,1),ep(1,2));
% 2/ asymptotic Kalman gain
pnam{1,2} = 'kinf';
pmin(1,2) = norminv(0.01,ep(2,1),ep(2,2));
pmax(1,2) = norminv(0.99,ep(2,1),ep(2,2));
pfun{1,2} = @(x)normpdf(x,ep(2,1),ep(2,2));
pini(1,2) = normstat(ep(2,1),ep(2,2));
pplb(1,2) = norminv(0.1587,ep(2,1),ep(2,2));
ppub(1,2) = norminv(0.8413,ep(2,1),ep(2,2));
% 3/ learning noise: scaling w/ prediction error
pnam{1,3} = 'zeta';
pmin(1,3) = norminv(0.01,ep(3,1),ep(3,2));
pmax(1,3) = norminv(0.99,ep(3,1),ep(3,2));
pfun{1,3} = @(x)normpdf(x,ep(3,1),ep(3,2));
pini(1,3) = normstat(ep(3,1),ep(3,2));
pplb(1,3) = norminv(0.1587,ep(3,1),ep(3,2));
ppub(1,3) = norminv(0.8413,ep(3,1),ep(3,2));
% 4/ learning noise: constant term - NOT USED
pnam{1,4} = 'ksi'; 
pmin(1,4) = 0.001;
pmax(1,4) = 1;
pfun{1,4} = @(x)gampdf(x,4,0.025);
pini(1,4) = gamstat(4,0.025);
pplb(1,4) = gaminv(0.1587,4,0.025);
ppub(1,4) = gaminv(0.8413,4,0.025);
% 5/ softmax temperature
switch cscheme
    case 'qvs' % Q-value sampling
        pnam{1,5} = 'theta';
        pmin(1,5) = 0.001;
        pmax(1,5) = 1;
        pfun{1,5} = @(x)gampdf(x,4,0.025);
        pini(1,5) = gamstat(4,0.025);
        pplb(1,5) = gaminv(0.1587,4,0.025);
        ppub(1,5) = gaminv(0.8413,4,0.025);
    case 'ths' % Thompson sampling
        pnam{1,5} = 'theta';
        pmin(1,5) = norminv(0.01,ep(4,1),ep(4,2));
        pmax(1,5) = norminv(0.99,ep(4,1),ep(4,2));
        pfun{1,5} = @(x)normpdf(x,ep(4,1),ep(4,2));
        pini(1,5) = normstat(ep(4,1),ep(4,2));
        pplb(1,5) = norminv(0.1587,ep(4,1),ep(4,2));
        ppub(1,5) = norminv(0.1587,ep(4,1),ep(4,2));
end
% 6/ fraction of trials with blind choices based on structure
pnam{1,6} = 'epsi';
pmin(1,6) = 0;
pmax(1,6) = 1;
pfun{1,6} = @(x)betapdf(x,1,1);
pini(1,6) = betastat(1,1);
pplb(1,6) = betainv(0.1587,1,1);
ppub(1,6) = betainv(0.8413,1,1);

% define fixed parameters
npar = numel(pnam);
pfix = cell(1,npar);
for i = 1:npar
    if isfield(cfg,pnam{i})
        pfix{i} = cfg.(pnam{i});
        % clip fixed parameters between minimum and maximum values
        pfix{i} = min(max(pfix{i},pmin(i)),pmax(i));
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

if nfit > 0
    
    % configure VBMC
    options = vbmc('defaults');
    options.MaxIter = 300; % maximum number of iterations
    options.MaxFunEvals = 500; % maximum number of function evaluations
    options.SpecifyTargetNoise = true; % noisy log-posterior function
    % set display level
    if verbose
        options.Display = 'iter';
    else
        options.Display = 'none';
    end
    
    % fit model using VBMC
    [vp,elbo,~,exitflag,output] = vbmc(@(x)fun(x), ...
        pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,options);
    
    % generate 10^6 samples from the variational posterior
    xsmp = vbmc_rnd(vp,1e6);
    
    xmap = vbmc_mode(vp); % posterior mode
    xavg = mean(xsmp,1); % posterior mean
    xstd = std(xsmp,[],1); % posterior s.d.
    xcov = cov(xsmp); % posterior covariance matrix
    xmed = median(xsmp,1); % posterior medians
    xiqr = quantile(xsmp,[0.25,0.75],1); % posterior interquartile ranges
    
    % create output structure with parameter values
    xhat = xmap; % use posterior mode
    phat = getpval(xhat);
    out = cell2struct(phat(:),pnam(:));
    
    % create substructures with parameter values
    phat_map = getpval(xmap); % posterior mode
    out.pmap = cell2struct(phat_map(:),pnam(:));
    phat_avg = getpval(xavg); % posterior mean
    out.pavg = cell2struct(phat_avg(:),pnam(:));
    
    % store fitting information
    out.nsmp = nsmp; % number of samples used by particle filter
    out.nres = nres; % number of re-samples for bootstrapping
    
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
    
    % store extra VBMC output
    out.exitflag = exitflag;
    out.output   = output;
    
else
    
    % create output structure with fixed parameter values
    xhat = [];
    phat = getpval(xhat);
    out = cell2struct(phat(:),pnam(:));
    
end

% get log-posterior and its bootstrap s.d. estimate
[out.l,out.l_sd] = fun(xhat);

% get information about variances used by Kalman filter
out.v0 = out.kini/(1-out.kini)*cfg.vs; % prior variance
out.vs = cfg.vs; % sampling variance
out.vd = fv(out.kinf); % diffusion variance
out.vr = log2(out.vd/out.vs); % their log-ratio

% store subject responses
out.resp = permute(reshape(resp,[nt,nb]),[2,1]);

% store filtered trajectories
[pt_hat,mt_hat,vt_hat] = getp(phat{:});
pt_hat = mean(pt_hat,2); % average across samples
mt_hat = mean(mt_hat,3); % average across samples
out.pt = permute(reshape(pt_hat,[nt,nb]),[2,1]);
out.mt = permute(reshape(mt_hat,[nt,nb,2]),[2,1,3]);
out.vt = permute(reshape(vt_hat,[nt,nb,2]),[2,1,3]);

    function [l,l_sd] = fun(p)
        % get parameter values
        pval = getpval(p);
        % get log-prior
        lp = 0;
        for k = 1:npar
            if isempty(pfix{k}) % free parameter
                lp = lp+log(pfun{k}(pval{k}));
            end
        end
        % get log-likelihood
        [ll,ll_sd] = getll(pval{:});
        % get log-posterior statistics
        l = ll+lp; % estimate
        l_sd = ll_sd; % bootstrap s.d. estimate
    end

    function [pval] = getpval(p)
        % parameter values
        pval = cell(1,npar);
        for k = 1:npar
            if isempty(pfix{k}) % free parameter
                pval{k} = p(ifit{k});
            else % fixed parameter
                pval{k} = pfix{k};
            end
        end
    end

    function [ll,ll_sd] = getll(varargin)
        % compute response probability
        p = getp(varargin{:});
        % compute bootstrap estimate of log-likelihood s.d.
        lres = nan(nres,1);
        for ires = 1:nres
            jres = randsample(nsmp,nsmp,true);
            pres = mean(p(:,jres),2);
            lres(ires) = ...
                sum(log(max(pres(resp == 1),minl)))+ ...
                sum(log(max(1-pres(resp == 2),minl)));
        end
        ll_sd = max(std(lres),1e-6);
        % compute log-likelihood estimate
        p = mean(p,2);
        ll = ...
            sum(log(max(p(resp == 1),minl)))+ ...
            sum(log(max(1-p(resp == 2),minl)));
    end

    function [pt,mt,vt] = getp(kini,kinf,zeta,ksi,theta,epsi)
        % retransform parameters from normally-distributed parameter to true parameter
        kini = 1/1+exp(-kini);
        kinf = 1/1+exp(-kinf);
        zeta = exp(zeta);
        theta = exp(theta);
        
        % express softmax temperature as selection noise
        ssel = pi/sqrt(6)*theta;
%        fprintf('epsi: %.04f | zeta: %.02f | kini: %.02f | kinf: %.02f | theta: %.02f\n', ...
%                epsi,zeta,kini,kinf,theta);        
        % run particle filter
        pt = nan(ntrl,nsmp); % response probabilities
        mt = nan(ntrl,2,nsmp); % posterior means
        vt = nan(ntrl,2); % posterior variances
        st = nan(2,nsmp); % filtering noise on current trial
        for itrl = 1:ntrl
            if trl(itrl) == 1
                % define structure bias for current block
                if sbias_cor
                    % toward correct action
                    rbias = 1;
                else
                    % toward first response
                    rbias = resp(itrl);
                end
                % initialize posterior means and variances
                if sbias_ini
                    % initialize biased means
                    mt(itrl,rbias,:) = ms;
                    mt(itrl,3-rbias,:) = 1-ms;
                else
                    % initialize unbiased means
                    mt(itrl,:,:) = 0.5;
                end
                vt(itrl,:) = kini/(1-kini)*vs;
                % choose first response
                if sbias_cor
                    % by sampling
                    md = reshape(mt(itrl,1,:)-mt(itrl,2,:),[1,nsmp]);
                    sd = sqrt(sum(vt(itrl,:)));
                    pd = 1-normcdf(0,md,sd);
                    pt(itrl,:) = (1-epsi)*pd+epsi;
                else
                    % same response as subject
                    pt(itrl,:) = rbias == 1;
                end
                continue
            end
            % compute Kalman gains
            kgain = vt(itrl-1,:)./(vt(itrl-1,:)+vs);
            % update posterior means and variances:
            c = resp(itrl-1);
            u = 3-c;
            % 1/ for chosen option
            mt(itrl,c,:) = mt(itrl-1,c,:)+kgain(c)*(rt(itrl-1)-mt(itrl-1,c,:));
            vt(itrl,c) = (1-kgain(c))*vt(itrl-1,c);
            switch nscheme
                case 'rpe' % noise scaling with reward prediction error
                    st(c,:) = sqrt(zeta^2*(rt(itrl-1)-mt(itrl-1,c,:)).^2+ksi^2);
                case 'upd' % noise scaling with value update
                    st(c,:) = sqrt(zeta^2*kgain(c)^2*(rt(itrl-1)-mt(itrl-1,c,:)).^2+ksi^2);
            end
            % 2/ for unchosen option:
            switch lscheme
                case 'ind' % assume independent action values
                    mt(itrl,u,:) = mt(itrl-1,u,:);
                    vt(itrl,u) = vt(itrl-1,u);
                    st(u,:) = ksi;
                case 'sym' % assume symmetric action values
                    mt(itrl,u,:) = mt(itrl-1,u,:)+kgain(u)*(1-rt(itrl-1)-mt(itrl-1,u,:));
                    vt(itrl,u) = (1-kgain(u))*vt(itrl-1,u);
                    switch nscheme
                        case 'rpe' % noise scaling with reward prediction error
                            st(u,:) = sqrt(zeta^2*(1-rt(itrl-1)-mt(itrl-1,u,:)).^2+ksi^2);
                        case 'upd' % noise scaling with value update
                            st(u,:) = sqrt(zeta^2*kgain(u)^2*(1-rt(itrl-1)-mt(itrl-1,u,:)).^2+ksi^2);
                    end
            end
            % account for diffusion process
            vt(itrl,:) = vt(itrl,:)+fv(kinf);
            % compute statistics of decision variable
            switch cscheme
                case 'qvs' % Q-value sampling
                    md = reshape(mt(itrl,1,:)-mt(itrl,2,:),[1,nsmp]);
                    sd = sqrt(sum(st.^2,1)+ssel^2);
                case 'ths' % Thompson sampling
                    md = reshape((mt(itrl,1,:)-mt(itrl,2,:))/sqrt(sum(vt(itrl,:))),[1,nsmp]);
                    sd = sqrt(sum(st.^2,1)/sum(vt(itrl,:))+ssel^2);
            end
            % sample trial types
            isl = rand(1,nsmp) < epsi; % choices based on structure learning
            irl = ~isl; % choices based on reinforcement learning
            % update trials with choices based on structure learning
            if nnz(isl) > 0
                % compute response probabilities
                pt(itrl,isl) = rbias == 1;
                % resample means without conditioning upon response
                mt(itrl,:,isl) = normrnd( ...
                    reshape(mt(itrl,:,isl),[2,nnz(isl)]), ...
                    st(:,isl));
            end
            % update trials with choices based on reinforcement learning
            if nnz(irl) > 0
                % compute response probabilities
                pt(itrl,irl) = 1-normcdf(0,md(irl),sd(irl));
                % resample means conditioned upon response
                if zeta > 0 || ksi > 0
                    switch cscheme
                        case 'qvs' % Q-value sampling
                            mt(itrl,:,irl) = resample( ...
                                reshape(mt(itrl,:,irl),[2,nnz(irl)]), ...
                                st(:,irl),ssel,resp(itrl));
                        case 'ths' % Thompson sampling
                            mt(itrl,:,irl) = resample( ...
                                reshape(mt(itrl,:,irl),[2,nnz(irl)]), ...
                                st(:,irl),ssel*sqrt(sum(vt(itrl,:))),resp(itrl));
                    end
                end
            end
        end
    end
end

function [xt] = resample(m,s,ssel,r)
% 1/ resample (x1-x2)
md = m(1,:)-m(2,:);
sd = sqrt(sum(s.^2,1));
td = tnormrnd(md,sqrt(sd.^2+ssel.^2),r); 
xd = normrnd( ...
    (ssel.^2.*md+sd.^2.*td)./(ssel.^2+sd.^2), ...
    sqrt(ssel.^2.*sd.^2./(ssel.^2+sd.^2)));
% 2/ resample x1 from (x1-x2)
ax = s(1,:).^2./sd.^2;
mx = m(1,:)-ax.*md;
sx = sqrt(s(1,:).^2-ax.^2.*sd.^2);
x1 = ax.*xd+normrnd(mx,sx);
% 3/ return x1 and x2 = x1-(x1-x2)
xt = cat(1,x1,x1-xd);
end

function [x] = tnormrnd(m,s,d)
% sample from truncated normal distribution
if d == 1
    x = +rpnormv(+m,s);
else
    x = -rpnormv(-m,s);
end
end