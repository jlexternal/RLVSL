function [out] = fit_noisyKF(cfg)

% get experiment data
resp = cfg.resp; % responses
rt   = cfg.rt; % rewards
nb = size(rt,1); % number of blocks
nt = size(rt,2); % number of trials per block
ntrl = nb*nt; % total number of trials

% reshape experiment data
resp = reshape(resp',[],1);
rt   = reshape(rt',[],1);
trl  = reshape(repmat(1:nt,[nb,1])',[],1);

% get model parameters
sbias_ini = cfg.sbias_ini; % initial structure bias flag
nscheme   = cfg.nscheme; % computation noise scheme
lscheme   = cfg.lscheme; % counterfactual learning scheme
cscheme   = cfg.cscheme; % choice sampling scheme

% check model parameters
if ~islogical(sbias_ini)
    error('Invalid intial structure bias flag!');
end
if ~ismember(nscheme,{'rpe','upd'})
    error('Undefined computation noise scheme!');
end
if ~ismember(lscheme,{'ind','sym'})
    error('Undefined counterfactual learning scheme!');
end
if ~ismember(cscheme,{'qvs','ths'})
    error('Undefined choice sampling scheme!');
end

% get sampling statistics
ms = cfg.ms; % sampling mean
vs = cfg.vs; % sampling variance

% get fitting parameters
nsmp     = cfg.nsmp;
nres     = cfg.nres;
verbose  = cfg.verbose; % display level

% reparameterization functions
fk = @(v)1./(1+exp(+0.4486-log2(v)*0.6282)).^0.5057;
kv = 0.001:0.001:0.999;
vv = arrayfun(@(k)fzero(@(v)fk(v)-k,2.^[-30,+30]),kv);
fv = @(k)interp1(kv,vv,k,'pchip');

% define model parameters
pnam = cell(1,5); % name
pmin = nan(1,5);  % minimum value
pmax = nan(1,5);  % maximum value
pini = nan(1,5);  % initial value
pfun = cell(1,5); % log-prior function
pplb = nan(1,5);  % plausible lower bound
ppub = nan(1,5);  % plausible upper bound
% 1/ initial Kalman gain
pnam{1} = 'kini';
pmin(1) = 0.001;
pmax(1) = 0.999;
pini(1) = 0.5;
pplb(1) = 0.1;
ppub(1) = 0.9;
pfun{1} = @(x)unifpdf(x,pmin(1),pmax(1));
% 2/ asymptotic Kalman gain
pnam{2} = 'kinf';
pmin(2) = 0.001;
pmax(2) = 0.999;
pini(2) = 0.5;
pplb(2) = 0.1;
ppub(2) = 0.9;
pfun{2} = @(x)unifpdf(x,pmin(2),pmax(2));
% 3/ learning noise
pnam{3} = 'zeta';
pmin(3) = 0;
pmax(3) = 5;
pini(3) = 0.5;
pplb(3) = 0.1;
ppub(3) = 1;
pfun{3} = @(x)unifpdf(x,pmin(3),pmax(3));
% 4/ softmax temperature
pnam{4} = 'theta';
switch cscheme
    case 'qvs'
        pmin(4) = 0;
        pmax(4) = 1;
        pini(4) = 0.1;
        pplb(4) = 0.01;
        ppub(4) = 0.5;
    case 'ths'
        pmin(4) = 0;
        pmax(4) = 10;
        pini(4) = 1;
        pplb(4) = 0.1;
        ppub(4) = 5;
end
pfun{4} = @(x)unifpdf(x,pmin(4),pmax(4));
% 5/ fraction of structure-biased responses
pnam{5} = 'epsi';
pmin(5) = 0;
pmax(5) = 1;
pini(5) = 0.5;
pplb(5) = 0.1;
ppub(5) = 0.9;
pfun{5} = @(x)unifpdf(x,pmin(5),pmax(5));

% set number of parameters
npar = numel(pnam);

% define fixed parameters
pfix = cell(1,npar);
for i = 1:npar
    if isfield(cfg,pnam{i}) && ~isempty(cfg.(pnam{i}))
        pfix{i} = cfg.(pnam{i});
    end
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

% set number of fitted parameters
nfit = length(pfit_ini);

% configure VBMC
options = vbmc('defaults');
options.MaxIter = 300; % maximum number of iterations
options.MaxFunEvals = 500; % maximum number of function evaluations
options.SpecifyTargetNoise = true; % noisy log-posterior function
switch verbose % display level
    case 0, options.Display = 'none';
    case 1, options.Display = 'final';
    case 2, options.Display = 'iter';
end

% fit model using VBMC
[vp,elbo,~,exitflag,output] = vbmc(@(x)getlp(x), ...
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
xhat = xmap; % use posterior mean
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

    function [pval] = getpval(p)
        % get parameter values
        pval = cell(1,npar);
        for k = 1:npar
            if isempty(pfix{k}) % free parameter
                pval{k} = p(ifit{k});
            else % fixed parameter
                pval{k} = pfix{k};
            end
        end
    end

    function [lp,lp_sd] = getlp(p)
        % get parameter values
        pval = getpval(p);
        % get log-prior
        l0 = 0;
        for k = 1:npar
            if isempty(pfix{k}) % free parameter
                l0 = l0+log(pfun{k}(pval{k}));
            end
        end
        % get log-likelihood
        [ll,ll_sd] = getll(pval{:});
        % get log-posterior
        lp = ll+l0; % estimate
        lp_sd = ll_sd; % bootstrap s.d.
    end

    function [negll] = getnegll(p)
        % get parameter values
        pval = getpval(p);
        % get negative log-likelihood
        negll = -getll(pval{:});
    end

    function [ll,ll_sd] = getll(varargin)
        % compute response probability
        p = getp(varargin{:});
        p = p*(1-1e-6)+0.5*(1e-6);
        if nargout > 1
            % compute bootstrap estimate of log-likelihood s.d.
            lres = nan(nres,1);
            for ires = 1:nres
                jres = randsample(nsmp,nsmp,true);
                pres = mean(p(:,jres),2);
                lres(ires) = ...
                    sum(log(pres(resp == 1)))+ ...
                    sum(log(1-pres(resp == 2)));
            end
            ll_sd = max(std(lres),1e-6);
        end
        % compute log-likelihood estimate
        p = mean(p,2);
        ll = ...
            sum(log(p(resp == 1)))+ ...
            sum(log(1-p(resp == 2)));
    end

    function [pt,mt,vt] = getp(kini,kinf,zeta,theta,epsi)
        % express softmax temperature as selection noise
        ssel = pi/sqrt(6)*theta;
        % run particle filter
        pt = nan(ntrl,nsmp); % response probabilities
        mt = nan(ntrl,2,nsmp); % posterior means
        vt = nan(ntrl,2); % posterior variances
        st = nan(2,nsmp); % filtering noise on current trial
        for itrl = 1:ntrl
            if trl(itrl) == 1
                % initialize posterior means and variances
                if sbias_ini
                    % initialize structure-biased means
                    rbias = resp(itrl);
                    mt(itrl,rbias,:) = ms;
                    mt(itrl,3-rbias,:) = 1-ms;
                else
                    % initialize structure-unbiased means
                    mt(itrl,:,:) = 0.5;
                end
                vt(itrl,:) = kini/(1-kini)*vs;
                % choose same first response as subject
                pt(itrl,:) = rbias == 1;
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
                    st(c,:) = zeta*abs(rt(itrl-1)-mt(itrl-1,c,:));
                case 'upd' % noise scaling with value update
                    st(c,:) = zeta*kgain(c)*abs(rt(itrl-1)-mt(itrl-1,c,:));
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
                            st(u,:) = zeta*abs(1-rt(itrl-1)-mt(itrl-1,u,:));
                        case 'upd' % noise scaling with value update
                            st(u,:) = zeta*kgain(u)*abs(1-rt(itrl-1)-mt(itrl-1,u,:));
                    end
            end
            % account for diffusion process
            vt(itrl,:) = vt(itrl,:)+fv(kinf);
            % compute statistics of decision variable
            switch cscheme
                case 'qvs' % q-value sampling or argmax
                    md = reshape(mt(itrl,1,:)-mt(itrl,2,:),[1,nsmp]);
                    sd = sqrt(sum(st.^2,1)+ssel^2);
                case 'ths' % Thompson sampling
                    vd = sum(vt(itrl,:));
                    md = reshape((mt(itrl,1,:)-mt(itrl,2,:))/sqrt(vd),[1,nsmp]);
                    sd = sqrt(sum(st.^2,1)/vd+ssel^2);
            end
            % compute response probabilities
            pt(itrl,:) = 1-normcdf(0,md,sd);
            pt(itrl,:) = pt(itrl,:)*(1-epsi)+(rbias == 1)*epsi;
            % update posterior means
            isl = rand(1,nsmp) < epsi; % choices based on structure learning
            irl = ~isl; % choices based on reinforcement learning
            if nnz(isl) > 0
                % filter posterior means not conditioned upon response
                mt(itrl,:,isl) = normrnd( ...
                    reshape(mt(itrl,:,isl),[2,nnz(isl)]), ...
                    st(:,isl));
            end
            if nnz(irl) > 0
                % filter posterior means conditioned upon response
                if zeta > 0
                    switch cscheme
                        case 'qvs' % q-value sampling or argmax
                            mt(itrl,:,irl) = filtcond( ...
                                reshape(mt(itrl,:,irl),[2,nnz(irl)]), ...
                                st(:,irl),ssel,0,resp(itrl));
                        case 'ths' % Thompson sampling
                            mt(itrl,:,irl) = filtcond( ...
                                reshape(mt(itrl,:,irl),[2,nnz(irl)]), ...
                                st(:,irl),ssel*sqrt(vd),0,resp(itrl));
                    end
                end
            end
            % clamp filtered posterior means between 0 and 1
            mt(itrl,:,:) = min(max(mt(itrl,:,:),0),1);
            % resample filtered posterior means wrt probability of response
            if resp(itrl) == 1
                wt = pt(itrl,:);
            else
                wt = 1-pt(itrl,:);
            end
            if any(wt > 0)
                mt(itrl,:,:) = mt(itrl,:,randsample(nsmp,nsmp,true,wt));
            end
        end
    end

end

function [xt] = filtcond(m,s,ssel,bsel,r)
% initialize output
n = size(m,2);
xt = nan(2,n);
% compute (x1-x2) statistics
md = m(1,:)-m(2,:); % mean
sd = sqrt(sum(s.^2,1)); % s.d.
% deal with samples with zero variance
i0 = sd == 0;
xt(:,i0) = m(:,i0);
% deal with samples with positive variance
m = m(:,~i0);
s = s(:,~i0);
md = md(:,~i0);
sd = sd(:,~i0);
if ~isscalar(ssel)
    ssel = ssel(~i0);
end
if ~isscalar(bsel)
    bsel = bsel(~i0);
end
% resample (x1-x2)
td = tnormrnd(md,sqrt(sd.^2+ssel.^2),-bsel,r); 
xd = normrnd( ...
    (ssel.^2.*md+sd.^2.*td)./(ssel.^2+sd.^2), ...
    sqrt(ssel.^2.*sd.^2./(ssel.^2+sd.^2)));
% resample x1 from (x1-x2)
ax = s(1,:).^2./sd.^2;
mx = m(1,:)-ax.*md;
sx = sqrt(s(1,:).^2-ax.^2.*sd.^2);
x1 = ax.*xd+normrnd(mx,sx);
% return x1 and x2 = x1-(x1-x2)
xt(:,~i0) = cat(1,x1,x1-xd);
end

function [x] = tnormrnd(m,s,t,d)
% sample from truncated normal distribution
if d == 1 % positive truncation
    x = +rpnormv(+m-t,s)+t;
else % negative truncation
    x = -rpnormv(-m+t,s)+t;
end
end