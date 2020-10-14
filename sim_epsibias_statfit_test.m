% sim_epsibias_statfit_test
%
% Testing: Simulation and recovery epsilon-greedy bias model for experiment RLVSL
%
% Jun Seok Lee <jlexternal@gmail.com>

clc;
clear all;
addpath('./vbmc')
addpath('./functions_local')
% Experimental parameters
nb = 16;
nt = 16;
% Generative parameters of winning distribution
% with false negative rate of 25%
ms = .55;
r_sd = .07413; 
vs   = r_sd^2; 

% Assumptions of the model
sbias_cor = false;   % 1st-choice bias toward the correct structure or subject (or random)
sbias_ini = false;   % KF means biased toward the correct structure
cscheme = 'qvs';    % 'arg'-argmax; 'qvs'-softmax;      'ths'-Thompson sampling
lscheme = 'sym';    % 'ind'-independent action values;  'sym'-symmetric action values
nscheme = 'upd';    % 'rpe'-noise scaling w/ RPE;       'upd'-noise scaling w/ value update
% Model parameters
ksi     = 0.0+eps;  % Learning noise constant

% Simulation settings
sameexpe = true;   % true if all sims see the same reward scheme
compexpe = []; % need to play with this once function is finished

% Run simulation test

% Organize parameter sets for simulation
epsis = 0.5; %linspace(0,.9,5);
zetas = 0.2;%[0:.1:.5]+eps;
kinis = [.9];%.5:.1:1-eps;
kinfs = [.1];%0:.1:.4+eps;
thetas = 0; %[0 .2 .4 .6 .8 1]+eps;
param_sets = {};

% define parameter sets
p_ctr = 0;
for epsi = epsis
    for zeta = zetas
        for kini = kinis
            for kinf = kinfs
                for theta = thetas
                    p_ctr = p_ctr + 1;
                    param_sets{p_ctr} = [epsi,zeta,kini,kinf,theta];
                end
            end
        end
    end
end


out_ctr = 0;
for ip = 1:numel(param_sets)
    fprintf('Simulating parameter set %d of %d\n',ip,numel(param_sets));
    epsi = param_sets{ip}(1);
    zeta = param_sets{ip}(2);
    kini = param_sets{ip}(3);
    kinf = param_sets{ip}(4);
    theta = param_sets{ip}(5);
    
    % configuration values
    cfg = struct; 
    cfg.nb = nb;    cfg.nt = nt; 
    cfg.ms = ms;    cfg.vs = vs; 
    cfg.sbias_cor = sbias_cor;  cfg.sbias_ini = sbias_ini;
    cfg.cscheme   = cscheme;    cfg.lscheme   = lscheme;    cfg.nscheme = nscheme;
    cfg.epsi = epsi; cfg.kini = kini; cfg.kinf = kinf; cfg.zeta = zeta; cfg.ksi = ksi; cfg.theta = theta;
    cfg.sameexpe    = sameexpe;
    cfg.ns = 1;
    
    % generate base sim
    sim_out = sim_epsibias_fn(cfg);
    
    cfg_p = struct;
    cfg_p.resp = sim_out.resp;
    pb = struct; % base proportions
    
    for type = {'correct','repprev','repfirst'}
        cfg_p.type = char(type);
        out_p1 = calc_prop_distr(cfg_p);
        switch cfg_p.type
            case 'correct'
                pb.p_c = out_p1.p;
            case 'repprev'
                pb.p_p = out_p1.p;
            case 'repfirst'
                pb.p_f = out_p1.p;
        end
    end
    
    
    % generate comparison sims
    cfg.firstresp   = sim_out.resp(:,1,:);    % vector of 1st-response for each block
    cfg.compexpe    = sim_out.rew;
    %cfg.zeta = .3;
    cfg.ns = 100;
    
    sim_comp = sim_epsibias_fn(cfg);
    
    cfg_p.resp = sim_comp.resp;
    cfg_p.type = 'correct';
    out_p2 = calc_prop_distr(cfg_p);
    p_i = [];
    
    for type = {'correct','repprev','repfirst'}
        cfg_p.type = char(type);
        out_p2 = calc_prop_distr(cfg_p);
        switch cfg_p.type
            case 'correct'
                p_i = cat(2,p_i,out_p2.p_i);
            case 'repprev'
                p_i = cat(2,p_i,out_p2.p_i);
            case 'repfirst'
                p_i = cat(2,p_i,out_p2.p_i);
        end
    end
    
    
    out_ctr = out_ctr + 1;
end

%% run recovery test

% inputs:   1/ raw rewards
%           2/ initial chosen values
%           3/ likelihood values to be compared 

cfg_fit.p_base = pb;
cfg_fit.rt = sim_out.rew;
cfg_fit.firstresp = sim_out.resp(:,1);
cfg_fit.ms = cfg.ms;
cfg_fit.vs = cfg.vs;
cfg_fit.nscheme = cfg.nscheme;
cfg_fit.lscheme = cfg.lscheme;
cfg_fit.cscheme = cfg.cscheme;
cfg_fit.sbias_ini = cfg.sbias_ini;
cfg_fit.sbias_cor = cfg.sbias_cor;
cfg_fit.verbose = true;
cfg_fit.ksi = 0;

out_fit = fit_stats_epsibias(cfg_fit); % fit the model to data

