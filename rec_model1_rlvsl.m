%function [params, llh] = rec_model1_rlvsl(cfg_tgt,cfg_mod,np)
%
% Parameter recovery script for model 1 on experiment RLVSL
% (4 parameter model with structure learning as updates of a Beta distribution)
%
% Inputs :
%           cfg_tgt : target distrib. config. structure 
%           cfg_mod : model config. structure
%
% Outputs :
%           params : recovered parameters 
%           llh    : log-likelihood from PF
clear all;


% Initialize random number generator
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

% Load the BADS files into working directory
addpath(genpath('./Toolboxes/bads-master'));

% Generative parameters

    % trial-to-trial RL difficulty calculation
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

cfg_tgt = struct;
cfg_tgt.mtgt = 5; % difference between the higher and lower shape value
cfg_tgt.stgt = sig_opti;

cfg_mod.nsims   = 1; % default should be 1
cfg_mod.beta    = 50;
cfg_mod.gamma   = .2;
cfg_mod.kappa   = .01;
cfg_mod.zeta    = .4;

fprintf('Generative parameters \n Rescaled Beta: %.2f, Gamma: %.2f, Kappa: %.4f, Zeta: %.2f\n', cfg_mod.beta/100,...
                                                                                                cfg_mod.gamma,...
                                                                                                cfg_mod.kappa,...
                                                                                                cfg_mod.zeta)

% Generate and store simulation data into structure
sim_struct = gen_sim_model1_rlvsl(cfg_tgt,cfg_mod);
global data;
data = sim_struct.sim_struct;

% Recovery parameters
global datasource;
global fitcond;
datasource  = 'sim';
fitcond     = {'rnd'}; % Specify the experimental condition {'rep','alt','rnd','all'}

switch fitcond{1}
    case 'rnd' % random across successive blocks
        ic = 3;
    case 'alt' % always alternating
        ic = 2;
    case 'rep' % always the same
        ic = 1;
end

% Initialize particle filter
global nparticles;
nparticles = 10000;
disp('########---------------------------------------########');
fprintf('Initializing %d particles...\n', nparticles);
fprintf('Searching parameter space on fitting condition: %s ...\n',fitcond{1});
disp('########---------------------------------------########');

% BADS parameters
%        beta/100 gamma  kappa  zeta

if strcmpi(fitcond{1},'rnd')
    % disregard fitting the gamma parameter
    % "fix" the softmax to argmax
    
    %startPt = [randi(50,1)*.01 .999 rand() rand()]; % vector corresponding to values of parameters to be estimated
    startPt = [.499 .999 rand() rand()]; % vector corresponding to values of parameters to be estimated
    lBound  = [.49 .99   0     0];     % HARD lower bound of parameter values
    uBound  = [.5   1   1     1];     % HARD upper bound of parameter values
    pLBound = [.49  .99   .001  .1];    % Plausible lower bound of parameter values (optional)
    pUBound = [.5   1   .05   .5];    % Plausible upper bound of parameter values (optional)
else
    startPt = [randi(50,1)*.01 rand() rand() rand()]; % vector corresponding to values of parameters to be estimated
                  %     Side note: should probably set this to various functions on the
                  %     parameter map to see if the optimum is found around the same
                  %     values on the noisy surface
    lBound  = [.49 0   0     0];     % HARD lower bound of parameter values
    uBound  = [.5   1   1     1];     % HARD upper bound of parameter values
    pLBound = [.49  .01 .001  .1];    % Plausible lower bound of parameter values (optional)
    pUBound = [.5   .5  .05   .5];    % Plausible upper bound of parameter values (optional)
end

%startPt = [cfg_mod.beta/100, cfg_mod.gamma, cfg_mod.kappa, cfg_mod.zeta]; % testing for optimizer rationale


% Changing parameter tolerance options on BADS
options = [];
options.TolMesh = 0.001;
options.UncertaintyHandling = 1; % stochastic objective function

% fit and see if recovered (Recall: BADS is a global minimum search algo)
[optiParams, objFn] = bads(@pf_model1_rlvsl, startPt, lBound, uBound, pLBound, pUBound, [], options)

% end % function disabled for testing purposes




