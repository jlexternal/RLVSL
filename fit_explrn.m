function [out] = fit_explrn(cfg)
%  FIT_EXPLRN  Fit exponential learning curve to data

% check input
if nargin < 1
    error('missing configuration structure!');
end
if ~all(isfield(cfg,{'trl','resp'}))
    error('invalid configuration structure!');
end
if ~isfield(cfg,'use_prf')
    cfg.use_prf = false;
end

% get data
trl  = cfg.trl(:); % trial number in block
resp = cfg.resp(:); % response (1:correct 2:error)
ntrl = numel(trl);

% use prior functions on parameter values?
use_prf = cfg.use_prf;

% compute lookup table for response probability array
iresp = sub2ind([ntrl,2],(1:ntrl)',resp);

% set ranges for parameter values
pfit_ini = [0.5;1  ;0.5]; % initial values
pfit_min = [0  ;0  ;0  ]; % minimum values
pfit_max = [1  ;1  ;100]; % maximum values
pfit_prf = { ... % prior functions:
    @(x)betapdf(x,1,1); ... % initial performance
    @(x)betapdf(x,1,1); ... % asymptotic performance
    @(x)gampdf(x,3,0.25)};  % learning rate

% fit exponential learning curve to data
pval = fmincon(@fmin,pfit_ini,[],[],[],[],pfit_min,pfit_max,[], ...
    optimset('Display','notify','FunValCheck','on', ...
    'Algorithm','interior-point','TolX',1e-20,'MaxFunEvals',1e6));

out = [];

% get best-fitting parameter values
out.pini = pval(1); % initial performance
out.pinf = pval(2); % asymptotic performance
out.lrat = pval(3); % learning rate

% get best-fitting learning curve
out.plrn = explrn(1:max(trl),out.pini,out.pinf,out.lrat);

    function [f] = fmin(p)
        % get log-likelihood
        f = getll(p);
        % add log-prior
        if use_prf
            for i = 1:3
                f = f+log(pfit_prf{i}(p(i)));
            end
        end
        % return negative log-likelihood
        f = -f;
    end

    function [ll] = getll(p)
        % get current parameter values
        pini = p(1); % initial performance
        pinf = p(2); % asymptotic performance
        lrat = p(3); % learning rate
        % compute response probabilities
        presp = nan(ntrl,2);
        presp(:,1) = explrn(trl,pini,pinf,lrat);
        presp(:,2) = 1-presp(:,1);
        % compute log-likelihood
        ll = sum(log(max(presp(iresp),1e-6)));
    end

end

function [p] = explrn(trl,pini,pinf,lrat)
% exponential learning function
B = (pinf-pini)*exp(lrat);
A = pinf-B;
p = A+B*(1-exp(-lrat*(trl)));
end
