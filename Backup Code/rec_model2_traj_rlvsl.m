%function [params, llh] = rec_model2_rlvsl(cfg_tgt,cfg_mod,np)
%
% Parameter recovery script for model 1 on experiment RLVSL
% (2 parameter model with structure learning fitted directly over each quarter)
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
addpath(genpath('./Toolboxes/vbmc'));

% Generative parameters

% trial-to-trial RL difficulty calculation
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

global cfg_tgt
cfg_tgt = struct;
cfg_tgt.mtgt    = 5;        % difference between the higher mean and 50
cfg_tgt.stgt    = sig_opti;

% Recovery parameters
global datasource;
global fitcond;
datasource  = 'hum';
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
nparticles = 1000;

global qrtr;
qrtr = 1;

global data;
if strcmpi(datasource,'sim')
    % Generate and store simulation data into structure
    sim_struct = gen_sim_model2_rlvsl(cfg_tgt,cfg_mod);
    data = sim_struct.sim_struct;
else
    subj = 3;
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',subj,subj)
    if ~exist(filename,'file')
        error('Missing experiment file!');
    end
    load(filename,'expe');
    data.expe = expe;
end



disp('########---------------------------------------########');
fprintf('Initializing %d particles...\n', nparticles);
fprintf('Fitting over experimental condition %s on quarter %d\n',fitcond{1},qrtr);
if strcmpi(datasource,'sim')
    fprintf('Generative parameters:\nKappa: %.2f, Zeta: %.2f, SL prior: %.2f\n',... 
                                                        cfg_mod.kappa,...
                                                        cfg_mod.zeta, ...
                                                        cfg_mod.strpr(ic,qrtr))
end
disp('########---------------------------------------########');
                                                             
% BADS parameters
%              kappa  zeta  sl_prior

if strcmpi(fitcond{1},'rnd')
    %startPt = [rand() rand() rand()]; % vector corresponding to values of parameters to be estimated
    startPt = [rand() rand() rand()]; % vector corresponding to values of parameters to be estimated

    lBound  = [0    0   .0001];     % HARD lower bound of parameter values
    uBound  = [1    1   .9999]; 	% HARD upper bound of parameter values
    pLBound = [.001 .01 .0001];     % Plausible lower bound of parameter values (optional)
    pUBound = [.49  .9  .9999];     % Plausible upper bound of parameter values (optional)
end


% Changing parameter tolerance options on BADS
options = [];
options.TolMesh = 0.001;
options.UncertaintyHandling = 1; % stochastic objective function
options.MaxFunEvals = (500);

% testing vbmc
nruns = 3;
vp = []; elbo = []; elbo_sd = []; exitflag = [];
for irun = 1:nruns
    [vp{irun},elbo(irun),elbo_sd(irun),exitflag(irun),output] = vbmc(@pf_model2_traj_rlvsl, startPt, lBound, uBound, pLBound, pUBound)
end
%{
% fit and see if recovered (Recall: BADS is a global minimum search algo)
[optiParams, objFn] = bads(@pf_model2_traj_rlvsl, startPt, lBound, uBound, pLBound, pUBound, [], options)

% end % function disabled for testing purposes
%}

%%

Xs = vbmc_rnd(vp{1},1e6);  % Generate samples from the variational posterior
% We compute the pdf of the approximate posterior on a 2-D grid
plot_lb = [0 0 0];
plot_ub = quantile(Xs,0.999);
x1 = linspace(plot_lb(1),plot_ub(1),200);
x2 = linspace(plot_lb(2),plot_ub(2),200);
x3 = .5014;
[xa,xb,xc] = meshgrid(x1,x2,x3);  % Build the grid
xx = [xa(:),xb(:),xc(:)];         % Convert grids to a vertical array of 2-D points
yy = real(vbmc_pdf(vp{1},xx));       % Compute PDF values on specified points
%%
% Plot approximate posterior pdf (works only in 1-D and 2-D)
surf(x1,x2,reshape(yy,[numel(x1),numel(x2)]),'EdgeColor','none');
xlabel('kappa');
ylabel('zeta');
zlabel('Approximate posterior pdf');
set(gca,'TickDir','out');
set(gcf,'Color','w');

