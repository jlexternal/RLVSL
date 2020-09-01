% sim_epsibias
%
% Simulate and recover epsilon-greedy bias model for experiment RLVSL
%
% Jun Seok Lee <jlexternal@gmail.com>

clc;
clear all;
% Experimental parameters
nb = 16;
nt = 16;
% Generative parameters of winning distribution
% with false negative rate of 25%
r_mu = .55;
r_sd = .07413; 
vs   = r_sd^2; 
% reparameterization functions
fk = @(v)1./(1+exp(+0.4486-log2(v/vs)*0.6282)).^0.5057;
fv = @(k)fzero(@(v)fk(v)-min(max(k,0.001),0.999),vs.*2.^[-30,+30]);

% Assumptions of the model
sbias_cor = false;   % 1st-choice bias toward the correct structure
sbias_ini = false;   % KF means biased toward the correct structure
cscheme = 'qvs';    % 'arg'-argmax; 'qvs'-softmax;      'ths'-Thompson sampling
lscheme = 'sym';    % 'ind'-independent action values;  'sym'-symmetric action values
nscheme = 'rpe';    % 'rpe'-noise scaling w/ RPE;       'upd'-noise scaling w/ value update
% Model parameters
ns      = 27;       % Number of simulated agents to generate per given parameter
ksi     = 0.0+eps;  % Learning noise constant

% Simulation settings
sameexpe = false;   % true if all sims see the same reward scheme
nexp     = 10;      % number of different reward schemes to try per given parameter set

sim_struct = struct;

% Run simulation
if sbias_cor
    disp('Assuming 1st-choice bias toward the correct structure!');
else
    disp('Assuming NO 1st-choice bias toward correct structure!');
end
if sbias_ini
    disp('Assuming initial bias of the mean toward the correct structure!');
else
    disp('Assuming no initial means bias!');
end
if ~ismember(cscheme,{'arg','qvs','ths'})
    error('Undefined or unrecognized choice sampling scheme!');
end

% Organize parameter sets for simulation
epsis = linspace(0,.9,5); % if set to 1, unrecoverable
zetas = [0:.1:.5]+eps;
kinis = .7:.1:1;
kinfs = 0:.1:.3;
thetas = 0;
param_sets = {};

% reparametrizing theta for the different choice schemes
switch cscheme
    case 'qvs' % regular softmax
        thetas = thetas;
    case 'ths' % Thompson sampling
        thetas = thetas*2;
end

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
    
    ssel = pi/sqrt(6)*theta;
    
    out_ctr = out_ctr + 1;
    % Generate experiment (reward scheme)
    cfg_gb = struct; cfg_gb.ntrls = nt; cfg_gb.mgen = r_mu; cfg_gb.sgen = r_sd; cfg_gb.nbout = nb;
    rew = []; % (nb,nt,ns)
    rew = cat(3,rew,round(gen_blck_rlvsl(cfg_gb),2));
    if sameexpe
        rew = cat(3,rew,repmat(rew(:,:,1),[1 1 ns-1]));
    else
        for issim = 1:ns-1
            rew = cat(3,rew,round(gen_blck_rlvsl(cfg_gb),2));
        end
    end
    rew_c = nan(size(rew));

    % Kalman Filter variables
    pt = nan(nb,nt,ns);     % response probability
    rt = nan(nb,nt,ns);     % actual responses
    mt = nan(nb,nt,2,ns);   % posterior mean
    vt = nan(nb,nt,2,ns);   % posterior variance
    st = nan(nb,nt,2,ns);   % current-trial filtering noise
    rb = nan(ns,1);         % response bias
    
    for ib = 1:nb
        for it = 1:nt
            if it == 1
                % determine structure bias
                if sbias_cor
                    % correct 1st choice
                    rb(:) = 1;
                else
                    % random 1st choice
                    rb = randi(2,1,ns);
                end
                % initialize KF means and variances
                if sbias_ini
                    % initial tracking mean biased toward generative mean
                    ind_rb = rb == 1;
                    mt(ib,it,1,ind_rb)  = r_mu;
                    mt(ib,it,1,~ind_rb) = 1-r_mu;
                    mt(ib,it,2,:) = 1-mt(ib,it,1,:);
                else
                    % initial tracking mean unbiased
                    mt(ib,it,:,:) = .5;
                end
                % initialize posterior variance based on initial kalman gain parameter
                vt(ib,it,:,:) = kini/(1-kini)*vs;
                % first trial response probability
                if sbias_cor
                    md = mt(ib,it,1,:)-mt(ib,it,2,:);
                    sd = sqrt(sum(vt(ib,it,:,:),3));
                    pd = 1-normcdf(0,md,sd);
                    pt(ib,it,:) = (1-epsi)*pd + epsi;
                else
                    pt(ib,it,:) = rb == 1;
                end
                % first trial response % 1/correct, 2/incorrect
                rt(ib,it,:) = round(pt(ib,it,:)); % argmax choice
                rt(rt==0) = 2;
                continue;
            end
            rb(:) = 1;
            % update Kalman gain
            kt = reshape(vt(ib,it-1,:,:)./(vt(ib,it-1,:,:)+vs),[2 ns]);
            % update posterior mean & variance
            for io = 1:2
                ind_c = find(rt(ib,it-1,:)==io); % index of sims
                c = io;     % chosen option
                u = 3-io;   % unchosen option
                if io == 1
                    rew_seen    = rew(ib,it-1,ind_c);
                    rew_unseen  = 1-rew(ib,it-1,ind_c);
                    if it == nt
                        rew_c(ib,it,ind_c) = rew(ib,it,ind_c);
                    end
                else
                    rew_seen    = 1-rew(ib,it-1,ind_c);
                    rew_unseen  = rew(ib,it-1,ind_c);
                    if it == nt
                        rew_c(ib,it,ind_c) = 1-rew(ib,it,ind_c);
                    end
                end
                rew_c(ib,it-1,ind_c) = rew_seen; % output for recovery procedure
                % update tracking values
                rew_seen    = reshape(rew_seen,size(mt(ib,it-1,c,ind_c)));
                rew_unseen  = reshape(rew_unseen,size(mt(ib,it-1,c,ind_c)));
                % 1/chosen option
                mt(ib,it,c,ind_c) = mt(ib,it-1,c,ind_c) + reshape(kt(c,ind_c),size(rew_seen)).*(rew_seen-mt(ib,it-1,c,ind_c));
                vt(ib,it,c,ind_c) = (1-reshape(kt(c,ind_c),size(rew_seen))).*vt(ib,it-1,c,ind_c);
                st(ib,it,c,ind_c) = sqrt(zeta^2*((rew_seen-mt(ib,it-1,c,ind_c)).^2+ksi^2)); % RPE-scaled learning noise
                % 2/unchosen option
                mt(ib,it,u,ind_c) = mt(ib,it-1,u,ind_c) + reshape(kt(u,ind_c),size(rew_unseen)).*(rew_unseen-mt(ib,it-1,u,ind_c));
                vt(ib,it,u,ind_c) = (1-reshape(kt(u,ind_c),size(rew_unseen))).*vt(ib,it-1,u,ind_c);
                st(ib,it,u,ind_c) = sqrt(zeta^2*((rew_unseen-mt(ib,it-1,u,ind_c)).^2+ksi^2));
            end
            % variance extrapolation/diffusion process 
            vt(ib,it,:,:)  = vt(ib,it,:,:)+fv(kinf); % covariance noise update
            
            % decision variable stats
            if ~strcmpi(cscheme,'ths')
                md = reshape(mt(ib,it,1,:)-mt(ib,it,2,:),[1 ns]);
                sd = reshape(sqrt(sum(st(ib,it,:,:).^2,3)+ssel^2),[1 ns]);
            else
                md = reshape((mt(ib,it,1,:)-mt(ib,it,2,:))./sqrt(sum(vt(ib,it,:,:))),[1 ns]);
                sd = reshape(sqrt(sum(st(ib,it,:,:).^2,3)./sum(vt(ib,it,:,:),3)+ssel^2),[1 ns]);
            end
            
            % sample trial types (based on epsi param)
            isl = rand(1,ns) < epsi;
            irl = ~isl;
             % Structure-utilising agents
            if nnz(isl) > 0
                pt(ib,it,isl) = rb(isl) == 1;
                % resampling w/o conditioning on response
                mt(ib,it,:,isl) = normrnd(mt(ib,it,:,isl),st(ib,it,:,isl));
            end
            % RL-utilising agents
            if nnz(irl) > 0
                pt(ib,it,irl) = 1-normcdf(0,md(irl),sd(irl));
            end
            % extract choice from probabilities
            rt(ib,it,:) = round(pt(ib,it,:));
            rt(rt==0) = 2;
           
        end
    end
    
    % output simulation data/cfg
    sim_struct(out_ctr).epsi = epsi;
    sim_struct(out_ctr).zeta = zeta;
    sim_struct(out_ctr).kini = kini;
    sim_struct(out_ctr).kinf = kinf;
    sim_struct(out_ctr).theta = theta;
    sim_struct(out_ctr).ksi  = ksi;
    sim_struct(out_ctr).ns   = ns;
    sim_struct(out_ctr).ms   = r_mu;
    sim_struct(out_ctr).vs   = vs;
    sim_struct(out_ctr).resp = rt;
    sim_struct(out_ctr).rew_seen = rew_c;
    sim_struct(out_ctr).sbias_cor   = sbias_cor;
    sim_struct(out_ctr).sbias_ini   = sbias_ini;
    sim_struct(out_ctr).cscheme     = cscheme;
    sim_struct(out_ctr).lscheme     = lscheme;
    sim_struct(out_ctr).nscheme     = nscheme;
    
    % plot simulation data (for sanity check/debug purposes)
    if false
        rt(rt==2)=0;
        figure(1);
        hold on;
        shadedErrorBar(1:nt,mean(mean(rt,3),1),std(mean(rt,3))/sqrt(ns),'lineprops',{'LineWidth',2+epsi},'patchSaturation',.1);
        ylim([0 1]);
        yline(.5,'--','HandleVisibility','off');
        xticks([1:4]*4);
        xlabel('trial number');
        ylabel('proportion correct');
        title(sprintf('Params: kini:%0.2f, kinf: %0.2f, zeta: %0.2f, theta: %0.2f, epsi: %0.2f\n nsims:%d',kini,kinf,zeta,theta,epsi,ns));
        ylim([.4 1]);
    end
end

sim_data = struct;
sim_data.sim_struct = sim_struct;
sim_data.gen_parset = param_sets;

clearvars -except sim_data
savename = sprintf('data_sim_epsibias_%s',datestr(now,'ddmmyyyy'));
save(savename,'sim_data');

%% test batch function

nbatch = 5;
fn_rec_batch_epsibias(1);
