function [out] = fit_stats_epsibias(cfg)

% check configuration structure
if ~isfield(cfg,'p_base')
    error('Missing base proportions!');
end
if ~isfield(cfg,'firstresp')
    error('Missing vector of first responses!');
end
if ~isfield(cfg,'rt')
    error('Missing rewards!');
end
if ~all(isfield(cfg,{'ms','vs'}))
    error('Missing sampling statistics!');
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
    cfg.nsmp = 1e2;
end
if ~isfield(cfg,'nres')
    cfg.nres = 1e2;
end
if ~isfield(cfg,'verbose')
    cfg.verbose = false;
end

% check reward scaling
if any(cfg.rt(:)) > 1
    error('Rewards should be scaled between 0 and 1!');
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

eps = 1e-6;
% get experiment data
rt   = cfg.rt; % rewards
pb   = cfg.p_base; % base proportions (p_c,p_p,p_f substructure values)

nb = size(rt,1); % number of blocks
nt = size(rt,2); % number of trials per block
ntrl = nb*nt; % total number of trials
nsmp = cfg.nsmp; % number of samples
nres = cfg.nres; % number of re-samples for bootstrapping
minl = 0.5/ntrl; % minimum value for filtered likelihoods
verbose = cfg.verbose; % display level

% get sampling statistics
ms = cfg.ms; % sampling mean
vs = cfg.vs; % sampling variance

% get model type
nscheme = cfg.nscheme; % noise scaling scheme ('rpe' or 'upd')
lscheme = cfg.lscheme; % latent learning scheme ('ind' or 'sym')
cscheme = cfg.cscheme; % choice sampling scheme ('qvs' or 'ths')
sbias_ini = cfg.sbias_ini; % structure bias on initial action values?
sbias_cor = cfg.sbias_cor; % structure bias toward correct action?

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
pplb(1,1) = betainv(0.1587,1,1);
ppub(1,1) = betainv(0.8413,1,1);
% 2/ asymptotic Kalman gain
pnam{1,2} = 'kinf';
pmin(1,2) = 0;
pmax(1,2) = 1;
pfun{1,2} = @(x)betapdf(x,1,1);
pini(1,2) = betastat(1,1);
pplb(1,2) = betainv(0.1587,1,1);
ppub(1,2) = betainv(0.8413,1,1);
% 3/ learning noise: scaling w/ prediction error
pnam{1,3} = 'zeta';
pmin(1,3) = 0.001;
pmax(1,3) = 2;
pfun{1,3} = @(x)gampdf(x,4,0.125);
pini(1,3) = gamstat(4,0.125);
pplb(1,3) = gaminv(0.1587,4,0.125);
ppub(1,3) = gaminv(0.8413,4,0.125);
% 4/ learning noise: constant term
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
        pmin(1,5) = 0.001;
        pmax(1,5) = 20;
        pfun{1,5} = @(x)gampdf(x,4,0.5);
        pini(1,5) = gamstat(4,0.5);
        pplb(1,5) = gaminv(0.1587,4,0.5);
        ppub(1,5) = gaminv(0.8413,4,0.5);
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
    options.RetryMaxFunEvals = options.MaxFunEvals;
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

% get log-posterior and its bootstrapped s.d. estimate
[out.l,out.l_sd] = fun(xhat);

% get information about variances used by Kalman filter
out.v0 = out.kini/(1-out.kini)*cfg.vs; % prior variance
out.vs = cfg.vs; % sampling variance
out.vd = fv(out.kinf); % diffusion variance
out.vr = log2(out.vd/out.vs); % their log-ratio

% store best-fitting likelihood and proportion curves
[l_hat,ppn_i_hat] = getp(phat{:});
out.l     = l_hat;
out.ppn_i = ppn_i_hat;

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
        [l,ppn_i] = getp(varargin{:});
        % compute log-likelihood estimate
        ll = sum(max(log(l),eps));
        
        p_x = [pb.p_c pb.p_p pb.p_f]; % vector of the base proportion
        % compute boostrapped log-likelihood 
        lres = nan(nres,1);
        for ires = 1:nres
            jres = randsample(nsmp,nsmp,true); % indices for bootstrapping 
            pres = mean(ppn_i(:,:,jres),3); % bootstrapped mean proportion
            sres = std(ppn_i(:,:,jres),1,3); % spread around bootstrapped mean
            lres(ires) = sum(log(normpdf(p_x,pres,sres))); % bootstrapped likelihood
        end
        ll_sd = max(std(lres),1e-6);
    end

    function [l,ppn_i] = getp(kini,kinf,zeta,ksi,theta,epsi)
        cfg.rt = rt;
        cfg.nb = nb;    
        cfg.nt = nt;
        cfg.epsi = epsi; 
        cfg.kini = kini; 
        cfg.kinf = kinf; 
        cfg.zeta = zeta; 
        cfg.ksi = ksi; 
        cfg.theta = theta;
        cfg.sameexpe = true;
        cfg.ns = cfg.nsmp;
        
        sim_out = sim_epsibias_fn(cfg); % simulate responses to generate distribution
        
        % calculate likelihood distribution curves for simulations
        l = []; % actual likelihoods
        ppn_i = []; % individual proportions for ll_sd bootstrapping
        
        cfg_p       = struct;
        cfg_p.resp  = sim_out.resp; % simulated responses
        
        ls_sim      = struct; % simulation distribution curves for the three criteria
        
        for type = {'correct','repprev','repfirst'}
            cfg_p.type = char(type);
            out_ps = calc_prop_distr(cfg_p); % mean and std of simulations
            switch cfg_p.type
                case 'correct'
                    ls_sim.p_c = out_ps.p;
                    ls_sim.s_c = out_ps.s;
                    l = cat(2,l,normpdf(pb.p_c,ls_sim.p_c,ls_sim.s_c+eps));
                    ppn_i = cat(2,ppn_i,out_ps.p_i);
                case 'repprev'
                    ls_sim.p_p = out_ps.p;
                    ls_sim.s_p = out_ps.s;
                    l = cat(2,l,normpdf(pb.p_p,ls_sim.p_p,ls_sim.s_p+eps));
                    ppn_i = cat(2,ppn_i,out_ps.p_i);
                case 'repfirst'
                    ls_sim.p_f = out_ps.p;
                    ls_sim.s_f = out_ps.s;
                    l = cat(2,l,normpdf(pb.p_f,ls_sim.p_f,ls_sim.s_f+eps));
                    ppn_i = cat(2,ppn_i,out_ps.p_i);
            end
        end
        
        % potential error catching
        if any(isnan(l))
            error('NaN found in likelihood vector!');
        elseif numel(l) ~= 3*nt-2
            error('Number of elements in likelihood vector %d does not match expectation (46)',numel(l));
        end
        
    end
end