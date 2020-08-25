function [out] = fit_noisyKF_epsibias(cfg)

% check configuration structure
if ~isfield(cfg,'rt')
    error('Missing rewards!');
end
if ~isfield(cfg,'resp')
    error('Missing responses!');
end
if ~isfield(cfg,'vs')
    error('Missing sampling variance!');
end
if ~isfield(cfg,'nsmp')
    error('Missing number of samples!');
end
if ~isfield(cfg,'lstruct')
    cfg.lstruct = 'ind'; % ind:independent or sym:symmetrical
    warning('Assuming independence between options.');
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

% get experiment data
resp = cfg.resp; % responses
rt   = cfg.rt; % rewards

nb = size(rt,1); % number of blocks
nt = size(rt,2); % number of trials per block
ntrl = nb*nt; % total number of trials

% get number of samples for particle filtering
nsmp = cfg.nsmp;

% reshape experiment data
resp = reshape(resp',[],1);
rt   = reshape(rt',[],1);
trl  = reshape(repmat(1:nt,[nb,1])',[],1);

% get sampling variance (scaling parameter)
vs = cfg.vs;

% get assumed latent structure
lstruct = cfg.lstruct;

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
pmin(1,1) = 0;
pmax(1,1) = 1;
pfun{1,1} = @(x)betapdf(x,1,1);
pini(1,1) = betastat(1,1);
pplb(1,1) = betainv(0.15,1,1);
ppub(1,1) = betainv(0.85,1,1);
% 2/ asymptotic Kalman gain
pnam{1,2} = 'kinf';
pmin(1,2) = 0;
pmax(1,2) = 1;
pfun{1,2} = @(x)betapdf(x,1,1);
pini(1,2) = betastat(1,1);
pplb(1,2) = betainv(0.15,1,1);
ppub(1,2) = betainv(0.85,1,1);
% 3/ learning noise: scaling w/ prediction error
pnam{1,3} = 'zeta';
pmin(1,3) = 0.001;
pmax(1,3) = 10;
pfun{1,3} = @(x)gampdf(x,4,0.125);
pini(1,3) = gamstat(4,0.125);
pplb(1,3) = gaminv(0.15,4,0.125);
ppub(1,3) = gaminv(0.85,4,0.125);
% 4/ learning noise: constant term
pnam{1,4} = 'ksi';
pmin(1,4) = 0.001;
pmax(1,4) = 10;
pfun{1,4} = @(x)gampdf(x,4,0.025);
pini(1,4) = gamstat(4,0.025);
pplb(1,4) = gaminv(0.15,4,0.025);
ppub(1,4) = gaminv(0.85,4,0.025);
% 5/ softmax temperature
pnam{1,5} = 'theta';
pmin(1,5) = 0.001;
pmax(1,5) = 10;
pfun{1,5} = @(x)gampdf(x,4,0.025);
pini(1,5) = gamstat(4,0.025);
pplb(1,5) = gaminv(0.15,4,0.025);
ppub(1,5) = gaminv(0.85,4,0.025);
% 6/ fraction of trials with choices based on structure
pnam{1,6} = 'epsi';
pmin(1,6) = 0;
pmax(1,6) = 1;
pfun{1,6} = @(x)betapdf(x,1,1);
pini(1,6) = betastat(1,1);
pplb(1,6) = betainv(0.15,1,1);
ppub(1,6) = betainv(0.85,1,1);

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

% configure VBMC
options = vbmc('defaults');
if cfg.verbose, opt_disp = 'iter'; else, opt_disp = 'notify'; end
options.Display = opt_disp; % level of display
options.MaxIter = 300; % maximum number of iterations
options.MaxFunEvals = 500; % maximum number of function evaluations
options.UncertaintyHandling = 1; % noisy log-posterior function

% fit model using VBMC
[vp,elbo,~,exitflag,output] = vbmc(@(x)fun(x),pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,options);

% generate 10^6 samples from the variational posterior
xsmp = vbmc_rnd(vp,1e6);

xmap = vbmc_mode(vp); % posterior mode
xavg = mean(xsmp,1); % posterior means
xstd = std(xsmp,[],1); % posterior s.d.
xcov = cov(xsmp); % posterior covariance matrix
xmed = median(xsmp,1); % posterior medians
xiqr = quantile(xsmp,[0.25,0.75],1); % posterior interquartile ranges

% create output structure with parameter values
[~,phat] = fun(xmap); % use posterior mode
out = cell2struct(phat(:),pnam(:));

% store assumed learning structure
out.lstruct = lstruct;

% get information about variances used by Kalman filter
out.v0 = out.kini/(1-out.kini)*cfg.vs; % prior variance
out.vs = cfg.vs; % sampling variance
out.vd = fv(out.kinf); % diffusion variance
out.vr = log2(out.vd/out.vs); % their log-ratio

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

% store configuration structure
out.cfg = cfg;

% store filtered trajectories
[~,pt_hat,mt_hat,vt_hat] = getll(phat{:});
out.pt = permute(reshape(pt_hat,[nt,nb]),[2,1]);
out.mt = permute(reshape(mt_hat,[nt,nb,2]),[2,1,3]);
out.vt = permute(reshape(vt_hat,[nt,nb,2]),[2,1,3]);

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

    function [ll,pt,mt,vt] = getll(kini,kinf,zeta,ksi,theta,epsi)
        % express softmax temperature as selection noise
        ssel = pi/sqrt(6)*theta;
        % run particle filter
        pt = nan(ntrl,nsmp); % response probabilities
        mt = nan(ntrl,2,nsmp); % posterior means
        vt = nan(ntrl,2); % posterior variances
        st = nan(2,nsmp); % filtering noise
        for itrl = 1:ntrl
            if trl(itrl) == 1
                % initialize posterior means and variances
                mt(itrl,:,:) = 0.5;
                vt(itrl,:) = kini/(1-kini)*vs;
                % choose like the subject
                rbias = resp(itrl);
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
            st(c,:) = sqrt(zeta^2*(rt(itrl-1)-mt(itrl-1,c,:)).^2+ksi^2);
            % 2/ for unchosen option
            switch lstruct
                case 'ind' % assume independence
                    mt(itrl,u,:) = mt(itrl-1,u,:);
                    vt(itrl,u) = vt(itrl-1,u);
                    st(u,:) = ksi;
                case 'sym' % assume reward symmetry
                    mt(itrl,u,:) = mt(itrl-1,u,:)+kgain(u)*(1-rt(itrl-1)-mt(itrl-1,u,:));
                    vt(itrl,u) = (1-kgain(u))*vt(itrl-1,u);
                    st(u,:) = sqrt(zeta^2*(1-rt(itrl-1)-mt(itrl-1,u,:)).^2+ksi^2);
            end
            % account for diffusion process
            vt(itrl,:) = vt(itrl,:)+fv(kinf);
            % compute statistics of decision variable
            md = reshape(mt(itrl,1,:)-mt(itrl,2,:),[1,nsmp]);
            sd = sqrt(sum(st.^2,1)+ssel^2);
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
                    mt(itrl,:,irl) = resample( ...
                        reshape(mt(itrl,:,irl),[2,nnz(irl)]), ...
                        st(:,irl),ssel,resp(itrl));
                end
            end
        end
        % average across samples
        pt = mean(pt,2);
        mt = mean(mt,3);
        % compute log-likelihood
        ep = 1e-6; % epsilon
        ll = sum(log((pt.*(resp == 1)+(1-pt).*(resp ~= 1))*(1-ep)+0.5*ep));
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