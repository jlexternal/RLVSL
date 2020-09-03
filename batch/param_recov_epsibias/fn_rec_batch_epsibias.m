function fn_rec_batch_epsibias(ibatch)

% Parameter recovery code (batch) for the epsilon-bias model for experiment RLVSL.
%
% Note: nbatch must be user-specified within the code before submitting to queue.
% 
% Requires: 1/ Simulated data set with proper configuration variables
%           2/ Toolbox folder in working directory w/ Rand and rpnormv.m
%           3/ gen_blck_rlvsl.m (to generate blocks)
%           4/ fit_noisyKF_epsibias.m
%           5/ VBMC (Acerbi 2020)

% ***** REQUIRED: Manual user input for proper functionality ***** %

    % NOTE: the name of the string must match the data file in 
    %       working directory
    load('data_sim_epsibias_03092020.mat'); % load sim data
    nbatch = 48; % number of total batches to be sent to SLURM; calculate this properly to distribute the output evenly
    
% **************************************************************** %

sim_struct = sim_data.sim_struct;
param_sets = sim_data.gen_parset;

if ~bsxfun(@eq,numel(sim_struct),numel(param_sets))
    error('Number of parameter sets does not match the number of simulation outputs!');
end
if ibatch > nbatch
    error('Batch number is greater than total number of designated batches!')
end
addpath('./vbmc');
addpath('./Toolboxes');


% holds the index range of the parameters for each batch
idx_batch   = nan(nbatch,2);
% number of parameter sets per batch
n_per_batch = floor(numel(param_sets)/nbatch);
% calculate parameter set index limits for each batch
for i = 1:nbatch
    idx_batch(i,:) = [1+(i-1)*n_per_batch i*n_per_batch];
    if i == nbatch
        if mod(numel(param_sets),nbatch) ~= 0
            idx_batch(i,:) = [1+i*n_per_batch numel(param_sets)];
        end
    end
end

% Run parameter recovery for the chosen batch
for ip = idx_batch(ibatch,1):idx_batch(ibatch,end)
    epsi = param_sets{ip}(1);
    zeta = param_sets{ip}(2);
    kini = param_sets{ip}(3);
    kinf = param_sets{ip}(4);
    theta = param_sets{ip}(5);
    
    for isim = 1:sim_struct(ip).ns
        cfg = [];
        cfg.resp = sim_struct(ip).resp(:,:,isim);
        cfg.rt = sim_struct(ip).rew_seen(:,:,isim);
        cfg.vs = sim_struct(ip).vs;
        cfg.ms = sim_struct(ip).ms;
        cfg.nsmp = 1e3;
        cfg.ksi = 0; % assume no constant term in learning noise
        cfg.cscheme = sim_struct(ip).cscheme;
        cfg.lscheme = sim_struct(ip).lscheme;
        cfg.nscheme = sim_struct(ip).nscheme;
        cfg.sbias_cor = sim_struct(ip).sbias_cor;
        cfg.sbias_ini = sim_struct(ip).sbias_ini;
        cfg.verbose = true;

        out_fit{ip,isim} = fit_noisyKF_epsibias(cfg); % fit the model
    end
end

savename = ['out_rec_epsibias_' num2str(nbatch) '_' num2str(ibatch)];
save(savename,'out_fit');

end