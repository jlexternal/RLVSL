% sim_batch_statfit_epsibias
%
% Generate data structure for recovery of epsilon-greedy bias model for experiment RLVSL
% with the statistical model fitter
%
% Jun Seok Lee <jlexternal@gmail.com>

clc;
clear all;
addpath('./vbmc');
% Experimental parameters
nb = 16;
nt = 16;
ns = 27; % number of unique datasets per given parameter set
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
sameexpe = false;   % true if all sims see the same reward scheme

% Run simulation test

% Organize parameter sets for simulation
epsis = linspace(0,.9,5)+eps; % if set to 1, unrecoverable
zetas = [0:.1:.5]+eps;
kinis = .7:.1:1-eps; % should never equal 1, otherwise vt blows up
kinfs = 0:.1:.3+eps;
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

sim_struct = struct;
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
    cfg.ns = ns;
    
    % generate base sim
    sim_struct_temp = sim_epsibias_fn(cfg);
    for fn = fieldnames(sim_struct_temp)'
        sim_struct(ip).(fn{1}) = sim_struct_temp.(fn{1});
    end
    
    % calculate the statistics for fit
    cfg_p = struct;
    cfg_p.resp = sim_struct.resp;
    pb = struct; % base proportions
    for type = {'correct','repprev','repfirst'}
        cfg_p.type = char(type);
        out_p1 = calc_prop_distr(cfg_p);
        switch cfg_p.type
            case 'correct'
                pb.p_c = out_p1.p_i;
            case 'repprev'
                pb.p_p = out_p1.p_i;
            case 'repfirst'
                pb.p_f = out_p1.p_i;
        end
    end
    
    % output simulation data/cfg
    sim_struct(ip).pb = pb;
end

clearvars -except sim_struct param_sets
savename = sprintf('data_sim_statfit_epsibias_%s',datestr(now,'ddmmyyyy'));
save(savename,'sim_struct','param_sets');

%% test batch function
fn_rec_batch_statfit_epsibias(38);

