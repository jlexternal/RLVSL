function [choices,block,q] = testcase_gen_block(cfg)
% TESTCASE GEN BLOCK Generates a block of trials based on configuration input
% 
% Inputs
%           cfg         : configuration structure
%           cfg.center  : center value between the two generative means
%           cfg.d       : radius (distance) of the two generative means from center
%           cfg.stdev   : standard deviation of the two generative distributions
%           cfg.ntrials : number of trials per block
%           cfg.beta    : inverse temperature parameter of softmax choice
% Outputs
%           choices     : model responses
%           block       : outcomes (both chosen and foregone)
%           q           : model estimates for the CORRECT option
%
%   This function generates a block of the partial outcome two-armed bandit task with 
%   stable means based on the configuration values indicated above.
% 
%   The function also outputs choices made by an optimal observer via the Kalman
%   filter method.
%
% Jun Seok Lee - Oct 2019

if ~all(isfield(cfg,{'d','stdev','ntrials','beta','center'})) %ensure complete cfg structure
    error('Necessary configuration fields are missing');
end

center  = cfg.center;   % intermediate value between the means of the two distributions
d       = cfg.d;        % distance from center
stdev   = cfg.stdev;    % standard deviation of distributions
ntrials = cfg.ntrials;  % number of trials 
beta    = cfg.beta;     % softmax inv. temp.

% Declare variables 
x       = [];   % observed outcomes | 2xNTRIALS array
choice  = [];   % optimal choices   | 1xNTRIALS array
estis   = [];   % estimations       | 2xNTRIALS array
k       = [];   % learning rates    | 2xNTRIALS array

% Initial filter and estimation variables
estis(1,1) = center;    % init estimation of choice 1 at center (flat prior)
estis(2,1) = center;    % init estimation of choice 2 at center (flat prior)
k(1,1)      = 0;        % init learning rate for bandit 1
k(2,1)      = 0;        % init learning rate for bandit 2
nu          = 0;        % init process uncertainty
omega       = stdev.^2; % init observation noise
w1          = stdev.^2; % init posterior variance for choice estimate 1
w2          = stdev.^2; % init posterior variance for choice estimate 2

for i = 1:ntrials
    % Choice step (softmax)
    prob1       = beta.*estis(1,i);
    prob2       = beta.*estis(2,i);
    weights     = [exp(prob1) exp(prob2)]./exp(prob1+prob2);
    choice(i)   = datasample([1 2], 1, 'Replace', true, 'Weights', weights); % 1-lower mean, 2-higher mean
    
    % Outcome sampling step
    if choice(i) == 1
        x(1,i) = round(normrnd(center-d,stdev));
        x(2,i) = round(normrnd(center+d,stdev));
    else
        x(1,i) = round(normrnd(center-d,stdev));
        x(2,i) = round(normrnd(center+d,stdev));
    end
    
    % Estimation step
    if choice(i) == 1
        estis(1,i+1)    = kalman(estis(1,i),k(1,i),x(1,i));
        estis(2,i+1)    = estis(2,i);                % propagate old estimate of option 2
    else
        estis(1,i+1)    = estis(1,i);                % propagate old estimate of option 1
        estis(2,i+1)    = kalman(estis(2,i),k(2,i),x(2,i));
    end
    
    % Filtering step
    if choice(i) == 1
        k(1,i+1)        = lrate(w1, nu, omega);
        w1              = (1-k(1,i+1)).*(w1+nu);
        k(2,i+1)        = k(2,i);
    else
        k(2,i+1)        = lrate(w2, nu, omega);
        w2              = (1-k(2,i+1)).*(w2+nu);
        k(1,i+1)        = k(1,i);
    end

end
choices = choice;
block = x;
q = estis(2,:);
end

% Local functions
function out = kalman(x,k,o) 
    % Kalman filter estimation for the mean
    %   inputs: (previous estimate, kalman gain, observation)
    out = x+k.*(o-x);
end
function out = lrate(w,nu,omega)
    % Learning rate update
    %   inputs: (posterior variance, process uncertainty, observation noise)
    out = (w+nu)./(w+nu+omega);
end
