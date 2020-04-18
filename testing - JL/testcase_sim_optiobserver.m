function [choices_out,estis_out,ks_out] = testcase_sim_optiobserver(cfg)
%
% Inputs
%           cfg             : configuration structure
%           cfg.nu          : process uncertainty
%           cfg.omega       : observation noise
%           cfg.w           : posterior variance(s)
%           cfg.beta        : softmax choice inverse temperature
%           cfg.block       : trials to be seen (partially) by optimal observer
%
%           cfg.priorest    : (optional) prior learned estimates (2x1 array)
%           cfg.priork      : (optional) prior learned learning rates (2x1 array)
% 
% Outputs
%           out             : output structure
%
%   This function simulates the optimal observer for a given block of the partial
%   outcome 2-armed bandit task.

if ~all(isfield(cfg,{'nu','omega','w','beta','block'})) %ensure complete cfg structure
    error('Necessary configuration fields are missing');
end
if ~isfield(cfg,'priorest')
    cfg.priorest = [50; 50]; % no prior
end

% tracking params
nu          = cfg.nu;   
omega       = cfg.omega;
w           = cfg.w;

if ~isfield(cfg,'priork')
    cfg.priork = [lrate(w(1,:),nu,omega); lrate(w(2,:),nu,omega)];      % initial learning rate
end

% softmax params
beta        = cfg.beta;
block       = cfg.block;
ntrials     = size(block,2);
% initialize variables
choice  = [];   % optimal choices   | 1xNTRIALS array
estis   = [];   % estimations       | 2xNTRIALS array
k       = [];   % learning rates    | 2xNTRIALS array
% initialize tracking variables
estis(:,1)  = cfg.priorest;
k(:,1)      = cfg.priork;

for i = 1:ntrials
    % make decision (softmax)
    prob1       = beta.*estis(1,i);
    prob2       = beta.*estis(2,i);
    weights     = [exp(prob1) exp(prob2)]./exp(prob1+prob2);
    choice(i)   = datasample([1 2], 1, 'Replace', true, 'Weights', weights); % 1-lower mean, 2-higher mean
    
    % Estimation update based on choice outcome
    if choice(i) == 1
        estis(1,i+1)    = kalman(estis(1,i),k(1,i),block(1,i));
        estis(2,i+1)    = estis(2,i);                % propagate old estimate of option 2
    else
        estis(1,i+1)    = estis(1,i);                % propagate old estimate of option 1
        estis(2,i+1)    = kalman(estis(2,i),k(2,i),block(2,i));
    end
    
    % Filtering step
    if choice(i) == 1
        %update chosen params
        k(1,i+1)        = lrate(w(1,i), nu, omega);
        w(1,i+1)        = (1-k(1,i+1)).*(w(1,i)+nu);
        %propagate unchosen params
        k(2,i+1)        = k(2,i);
        w(2,i+1)        = w(2,i);
    else
        %update chosen params
        k(2,i+1)        = lrate(w(2,i), nu, omega);
        w(2,i+1)        = (1-k(2,i+1)).*(w(2,i)+nu);
        %propagate unchosen params
        k(1,i+1)        = k(1,i);
        w(1,i+1)        = w(1,i);
    end
end
choices_out = choice;
estis_out   = estis;
ks_out      = k;
end

% Local functions
function out = kalman(x,k,o) 
    % Kalman filter estimation for the mean
    %   inputs: (previous estimate, kalman gain, observation)
    out = x+k.*(o-x);
end
function out= lrate(w,nu,omega)
    % Learning rate update
    %   inputs: (posterior variance, process uncertainty, observation noise)
    out = (w+nu)./(w+nu+omega);
end
