% run_rec_RLpSL2_rlvsl

clear all
close all
clc

% add VBMC toolbox to path
addpath('./vbmc');

% quarters to fit
%quar2fit = 1:4;

model_cfg = struct;
% model configuration structure values 
nsims   = 1;
zetas  	= [0 .2 .4]; %linspace(.2,.7,5);
alphas  = [0 .2 .4]; %linspace(0,.5,5);
strprs 	= [.5 .7 .8]; %linspace(.4,.8,5);
nfits   = numel(zetas)*numel(alphas)*numel(strprs);

out_fit  = cell(nfits,1);
out_sim  = cell(nfits,1);

% trial-to-trial RL difficulty calculation
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

isim = 0;
for izeta = zetas
    for ialpha = alphas
        for iprior = strprs
            isim = isim + 1;
            model_cfg.nsims = nsims;
            model_cfg.zeta  = izeta;
            model_cfg.alpha = ialpha;
            model_cfg.strpr = iprior;
            
            target_cfg = struct;
            target_cfg.mtgt = 5;        % difference between high target and center value
            target_cfg.stgt = sig_opti;

            % generate model simulations
            out_sim{isim} = gen_sim_model2_rlvsl(target_cfg,model_cfg);
        end
    end
end


%% organize response and outcome data structures for fitting function and recover
nb      = 16;
nt      = 16;
cond    = 3; % consider only the random condition since the model doesn't distinguish them

resp_struct = struct;
for ifit = 1:nfits
    
    % extract responses and outcomes
    choices = zeros(nb,nt);
    outcome = zeros(nb,nt);

    condtype = 'rnd';
    icond    = 3;
    
    % aggregate all blocks of a specific condition 
    ib_c = 0;
    for ib = 1:out_sim{ifit}.sim_struct.expe(1).cfg.nbout
        ctype = out_sim{ifit}.sim_struct.expe(ib).type;
        if ismember(ctype,condtype)
            ib_c = ib_c + 1;
            choices(ib_c,:) = out_sim{ifit}.sim_struct.expe(ib).resp;
            outcome(ib_c,:) = out_sim{ifit}.sim_struct.expe(ib).blck_trn+50;
        else

        end
    end

    resp_struct(ifit).outcome = outcome;
    resp_struct(ifit).choices = choices;
    
    % set up structures for fitter
    blk = kron(1:nb,ones(1,nt))';
    trl = repmat((1:nt)',[nb,1]);

    resp = resp_struct(ifit).choices(:,:)';
    resp = resp(:); % vectorize response matrix

    rew = nan(size(resp,1),2); % rewards of chosen & unchosen options
    for i = 1:size(resp,1) % pointers on blk and trl

        rew(i,resp(i))   = resp_struct(ifit).outcome(blk(i),trl(i))/100; % chosen
        rew(i,3-resp(i)) = 1-rew(i,resp(i));                                   % unchosen
    end

    % instantiate configuration structure
    cfg         = [];
    cfg.icond   = icond; % condition type
    cfg.nsmp    = 1e3;   % number of samples
    cfg.verbose = true;  % verbose VBMC output
    cfg.stgt    = sig_opti/100;
    %cfg.prior = .5; %test

    % fixed parameters
    cfg.tau     = 1e-6; % assume argmax choice policy
    cfg.ksi     = 1e-6; % assume pure Weber noise (no constant term)

    
    iz = out_sim{ifit}.sim_struct.sim.cfg_model.zeta;
    ia = out_sim{ifit}.sim_struct.sim.cfg_model.alpha;
    ip = out_sim{ifit}.sim_struct.sim.cfg_model.strpr;
    
    % fit on quarters (in non-random conditions)
    %for iq = quar2fit
        % to be fitted
            % zeta  - scaling of learning noise with prediction error
            % alpha - KF learning rate asymptote
            % prior - strength of the prior belief on the correct option
            
        %fprintf('Fitting model simulation over quarter %d\n',iq);

        %cfg.iquarter = iq;

        % chunk data structures into quarters
        blkstart  = 1;%4*(iq-1)+1;
        blkend    = 16;%blkstart+3;

        idx     = ismember(blk,blkstart:blkend);    % blocks corresponding to the quarter

        qresp   = resp.*idx;
        qrew    = rew.*(idx.*ones(size(idx,1),2));
        qtrl    = trl.*idx;
        qresp   = nonzeros(qresp);
        qrew    = [nonzeros(qrew(:,1)) nonzeros(qrew(:,2))];
        qtrl    = nonzeros(qtrl);

        cfg.resp    = qresp;
        cfg.rew     = qrew;
        cfg.trl     = qtrl;

        out_fit{ifit} = fit_RLpSL2_rlvsl(cfg);
        
        rz = out_fit{ifit}.zeta;
        ra = out_fit{ifit}.alpha;
        rp = out_fit{ifit}.priorpar;
        
        fprintf('Generative params: zeta: %.2f, alpha: %.2f, prior: %.2f\n',iz,ia,ip);
        fprintf('Recovered params: zeta: %.2f, alpha: %.2f, prior: %.2f\n',rz,ra,rp);
    %end
end
