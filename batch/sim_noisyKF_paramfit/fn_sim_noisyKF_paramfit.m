function fn_sim_noisyKF_paramfit(isubj)

% fn_sim_noisyKF_paramfit
%
% Objective: Simulate the noisyKF model with parameters sampled from the posterior
%             distribution from the fitting procedure on subjects.
%
% Jun Seok Lee <jlexternal@gmail.com>


% function based on subject (3cond x 4 quar = 12 minutes per subject)

nbatch = 28;

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);

% ensure proper pathing and function availability
addpath('./Toolboxes/Rand')
addpath('./vbmc')
addpath('./vbmc/utils')

% load subject data and fit structures
load('subj_resp_rew_all.mat','subj_resp_rew_all')
load('out_fit_all','out_fit_all')

% experimental parameters
cfg = struct;
cfg.nt = 16;
cfg.ms = .55;   cfg.vs = .07413^2; 
cfg.sbias_cor = false;  
cfg.sbias_ini = true;
cfg.cscheme = 'ths';  cfg.lscheme = 'sym';  cfg.nscheme = 'upd';
cfg.ns      = 2500; % 100 simulations take around 35 seconds per subject 
cfg.ksi     = 0;
cfg.epsi    = 0;
cfg.sameexpe = true;    % true if all sims see the same reward scheme

resp_sim = nan(4,16,cfg.ns,3,4,nbatch); % block, trial, nsims, condition, quarter, subject

issampling = false;
fprintf('Simulating subject %d...\n',isubj)
isubj_abs = subjlist(isubj);


for icond = 1:3
    fprintf('  Simulating condition %d...\n',icond)
    cfg.nb = 4;
    for iq = 1:5
        if iq < 5
            cfg.nb = 4;
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
        else
            if icond ~= 3
                continue
            else
                cfg.nb = 16;
                blockrange = 1:16;
            end
        end
        cfg.compexpe    = subj_resp_rew_all(isubj_abs).rew_expe(blockrange,:,icond)/100;
        cfg.firstresp   = subj_resp_rew_all(isubj_abs).resp(blockrange,1,icond);
        
        if issampling
            if icond ~= 3
                params_smpd = vbmc_rnd(out_fit_all{icond,iq,isubj}.vp,cfg.ns);
            else
                params_smpd = vbmc_rnd(out_fit_all{icond,5,isubj}.vp,cfg.ns);
            end
            cfg.kini    = params_smpd(:,1);
            cfg.kinf    = params_smpd(:,2);
            cfg.zeta    = params_smpd(:,3);
            cfg.theta   = params_smpd(:,4);
        else
            % use xmap (maximum a posteriori estimate)
            if icond ~= 3
                params_smpd = out_fit_all{icond,iq,isubj}.xmap;
            else
                params_smpd = out_fit_all{icond,5,isubj}.xmap;
            end
            cfg.kini    = params_smpd(1)*ones(cfg.ns,1);
            cfg.kinf    = params_smpd(2)*ones(cfg.ns,1);
            cfg.zeta    = params_smpd(3)*ones(cfg.ns,1);
            cfg.theta   = params_smpd(4)*ones(cfg.ns,1);
        end

        sim_out = sim_noisyKF_fn(cfg);

        if iq < 5
            resp_sim(:,:,:,icond,iq,isubj) = sim_out.resp;
        else
            for jq = 1:4
                blockrange = 4*(jq-1)+1:4*(jq-1)+4;
                resp_sim(:,:,:,icond,jq,isubj) = sim_out.resp(blockrange,:,:);
            end
        end
    end
end

savename = sprintf('out_resp_sim_noisyKF_%02d_%02d',nbatch,isubj);
save(savename,'resp_sim');

end
