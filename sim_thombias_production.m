% sim_thombias_production
%
% Simulate and recover Thompson-sampling bias model for experiment RLVSL
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
sbias_cor = false;  % 1st-choice bias toward the correct structure
sbias_ini = false;  % KF means biased toward the correct structure


% Model parameters
ns      = 27;       % Number of simulated agents to generate per given parameter
% KF learning parameters
kini    = 0.8-eps;  % Initial Kalman gain
kinf    = 0.3+eps;  % Asymptotic Kalman gain
% Learning noise parameters
zeta    = 0.3+eps;  % Learning noise scale
ksi     = 0.0+eps;  % Learning noise constant

%%%% Need to play with this a bit and figure out how to parametrize it
% Thompson-sampling bias parameter
delta   = 0.2;      % 

%%%%

% Selection parameters
theta   = 1;       % Inverse softmax temperature 

params_gen = struct;
params_gen.delta = delta;
params_gen.kini = kini;
params_gen.kinf = kinf;
params_gen.zeta = zeta;
params_gen.ksi  = ksi;

% Simulation settings
sameexpe = false;   % true if all sims see the same reward scheme

sim_struct = struct;

%% Run simulation
if sbias_cor
    disp('Assuming 1st-choice bias toward the correct structure!');
else
    disp('Assuming NO 1st-choice bias toward correct structure!');
end
if sbias_ini
    disp('Assuming initial bias of the mean toward the correct structure!');
else
    disp('Assuming NO initial means bias!');
end

% Organize parameter values into sets for simulation
deltas = .05; %linspace(0,0.2,4);
zetas = .3; %[0:.1:.4]+1e-6;
kinis = [0.90];%.5:.1:1;
kinfs = [0.10];%0:.1:.4;
thetas = 0; %[0 .2 .4 .6 1 ]+eps;
param_sets = {};

% define parameter sets
p_ctr = 0;
for delta = deltas
    for zeta = zetas
        for kini = kinis
            for kinf = kinfs
                for theta = thetas
                    p_ctr = p_ctr + 1;
                    param_sets{p_ctr} = [delta,zeta,kini,kinf,theta];
                end
            end
        end
    end
end

out_ctr = 0;
for ip = 1:numel(param_sets)
    fprintf('Simulating parameter set %d of %d\n',ip,numel(param_sets));
    delta = param_sets{ip}(1);
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
                    % random sampling of 1st choice based on structure bias strength
  %                  rb = randsample(2,ns,true,[.5+delta .5-delta]);
                    
                    % sampling based on argmax of drawn values
                    rb = double(normrnd(.5+delta,r_sd,ns,1) >= .5);
                    rb(rb~=1) = 2;
                end
                ind_rb = rb == 1;
                % initialize KF means and variances
                if sbias_ini
                    % initial tracking mean biased based on the delta parameter
                    mt(ib,it,1,ind_rb)  = .5+delta;
                    mt(ib,it,1,~ind_rb) = .5-delta;
                    mt(ib,it,2,:) = 1-mt(ib,it,1,:);
                else
                    % initial tracking mean unbiased
                    mt(ib,it,:,:) = .5;
                end
                % initialize posterior variance based on initial kalman gain parameter
                vt(ib,it,:,:) = kini/(1-kini)*vs;
                % first trial response probability
                if sbias_cor
                    md = (mt(ib,it,1,:)+delta)-mt(ib,it,2,:); % biasing the means via delta
                    sd = sqrt(sum(vt(ib,it,:,:),3));
                    pd = 1-normcdf(0,md,sd);
                    pt(ib,it,:) = pd;
                else
                    pt(ib,it,:) = rb == 1;
                end
                % first trial response % 1/correct, 2/incorrect
                rt(ib,it,:) = round(pt(ib,it,:)); % argmax choice
                rt(rt==0) = 2;
                continue;
            end
%            rb(:) = 1; % debug (ignore)

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
            md = reshape(((mt(ib,it,1,:)+delta)-mt(ib,it,2,:))./sqrt(sum(vt(ib,it,:,:))),[1 ns]);
            sd = reshape(sqrt(sum(st(ib,it,:,:).^2,3)./sum(vt(ib,it,:,:),3)+ssel^2),[1 ns]);
            % response probability
            pt(ib,it,:) = 1-normcdf(0,md,sd);
            % extract choice from probabilities
            rt(ib,it,:) = round(pt(ib,it,:));
            rt(rt==0) = 2;
            % resampling
            if zeta > 0 || ksi > 0
                delta_vec1 = reshape(double(ind_rb),size(mt(ib,it,1,:)));
                delta_vec2 = reshape(double(~ind_rb),size(mt(ib,it,1,:)));
                %mt(ib,it,:,:) = resample(reshape([mt(ib,it,1,:)+delta mt(ib,it,2,:)],[2,ns]),...
                mt(ib,it,:,:) = resample(reshape([mt(ib,it,1,:)+delta*delta_vec1 mt(ib,it,2,:)+delta*delta_vec2],[2,ns]),...
                                         reshape(st(ib,it,:,:),[2,ns]),...
                                         reshape(ssel*sqrt(sum(vt(ib,it,:,:),3)),[1 ns]),... % width increase due to KF variance
                                         reshape(rt(ib,it,:),[1 ns]));
            end
        end
    end
    
    % output simulation data
    sim_struct(out_ctr).delta = delta;
    sim_struct(out_ctr).zeta = zeta;
    sim_struct(out_ctr).kini = kini;
    sim_struct(out_ctr).kinf = kinf;
    sim_struct(out_ctr).theta = theta;
    sim_struct(out_ctr).ksi  = ksi;
    sim_struct(out_ctr).vs   = vs;
    sim_struct(out_ctr).resp = rt;
    sim_struct(out_ctr).rew_seen = rew_c;
    
    % plot simulation data
    if true
        rt(rt==2)=0;
        figure(1);
        hold on;
        shadedErrorBar(1:nt,mean(mean(rt,3),1),std(mean(rt,3))/sqrt(ns),'lineprops',{'LineWidth',2},'patchSaturation',.1);
        ylim([0 1]);
        yline(.5,'--','HandleVisibility','off');
        xticks([1:4]*4);
        xlabel('trial number');
        ylabel('proportion correct');
        title(sprintf('Params: kini:%0.2f, kinf: %0.2f, zeta: %0.2f, theta: %0.2f, delta: %0.2f\n nsims:%d',kini,kinf,zeta,theta,delta,ns));
    end
end
%clearvars -except param_sets sim_struct ns

%% Parameter recovery in BATCHES (divide parameter sets into separate "jobs")
if ~bsxfun(@eq,numel(sim_struct),numel(param_sets))
    error('Number of parameter sets does not match the number of simulation outputs!');
end
addpath('./vbmc');

nbatch     = 1; % number of batches 
% holds the index range of the parameters for each batch
idx_batch   = nan(nbatch,2);
% number of parameter sets per batch
n_per_batch = floor(numel(param_sets)/nbatch); 
% calculate parameter set index limits for each batch
for ibatch = 1:nbatch
    idx_batch(ibatch,:) = [1+(ibatch-1)*n_per_batch ibatch*n_per_batch];
    if ibatch == nbatch
        if mod(numel(param_sets),nbatch) ~= 0
            idx_batch(ibatch,:) = [1+ibatch*n_per_batch numel(param_sets)];
        end
    end
end

ibatch_run = 1; % choose which batch to recover
if ibatch_run > nbatch
    error('The chosen batch number exceeds the number of batches defined!')
end
% Run parameter recovery for the chosen batch
ibatch = 1;
for ip = idx_batch(ibatch,:)
    delta = param_sets{ip}(1);
    zeta = param_sets{ip}(2);
    kini = param_sets{ip}(3);
    kinf = param_sets{ip}(4);
    theta = param_sets{ip}(5);
    
    for isim = 1:ns
        
        fprintf('Generative parameters: delta: %.04f | zeta: %.02f | kini: %.02f | kinf: %.02f | theta: %.02f\n', ...
                delta,zeta,kini,kinf,theta);
        cfg = [];
        cfg.resp = sim_struct(ip).resp(:,:,isim);
        cfg.rt = sim_struct(ip).rew_seen(:,:,isim);
        cfg.ms = .55;
        cfg.vs = sim_struct(ip).vs;
        cfg.nsmp = 1e3;
        cfg.lstruct = 'sym'; % assume symmetric action values
        cfg.sbias_cor = false;
        cfg.verbose = true; % plot fitting information
        cfg.ksi = 0; % assume no constant term in learning noise
        cfg.theta = 0; % fix to argmax choice

        out_fit{ip,isim} = fit_noisyKF_thombias(cfg); % fit the model to data
        
        delta_fit{ip,isim} = out_fit{ip,isim}.delta;
        zeta_fit{ip,isim} = out_fit{ip,isim}.zeta;
        kini_fit{ip,isim} = out_fit{ip,isim}.kini;
        kinf_fit{ip,isim} = out_fit{ip,isim}.kinf;
        
        delta_rp = .4*delta_fit{ip,isim}-.2;% reparametrized from fitted transform delta
        
        fprintf('Recovered parameters: delta: %.04f | zeta: %.02f | kini: %.02f | kinf: %.02f | theta: %.02f\n', ...
                delta_rp,zeta_fit{ip,isim},kini_fit{ip,isim},kinf_fit{ip,isim},cfg.theta);
    end
end

%% Parameter recovery (fit model to simulated data)

if ~bsxfun(@eq,numel(sim_struct),numel(param_sets))
    error('Number of parameter sets does not match the number of simulation outputs!');
end
addpath('./vbmc')
for ip = 1:numel(param_sets)
    delta = param_sets{ip}(1);
    zeta = param_sets{ip}(2);
    kini = param_sets{ip}(3);
    kinf = param_sets{ip}(4);
    
    for isim = 1:ns
        cfg = [];
        cfg.resp = sim_struct(ip).resp(:,:,isim);
        cfg.rt = sim_struct(ip).rew_seen(:,:,isim);
        cfg.ms = .55;
        cfg.vs = sim_struct(ip).vs;
        cfg.nsmp = 1e3;
        cfg.lstruct = 'sym'; % assume symmetric action values\
        cfg.sbias_cor = false;
        cfg.verbose = true; % plot fitting information
        cfg.ksi = 0; % assume no constant term in learning noise

        out_fit{ip,isim} = fit_noisyKF_epsibias(cfg); % fit the model to data
        
        delta_fit{ip,isim} = out_fit{ip,isim}.delta;
        zeta_fit{ip,isim} = out_fit{ip,isim}.zeta;
        kini_fit{ip,isim} = out_fit{ip,isim}.kini;
        kinf_fit{ip,isim} = out_fit{ip,isim}.kinf;
    end
end
savename = ['fit_struct_epsibias_' datestr(now,'ddmmyyyy')];
save(savename,'out_fit','param_sets','sim_struct','epsi_fit','zeta_fit','kini_fit','kinf_fit');

%% Organize fit parameters

for ip = 1:numel(param_sets)
    % generative parameters
    param_gen(1,ip) = param_sets{ip}(1);
    param_gen(2,ip) = param_sets{ip}(2);
    param_gen(3,ip) = param_sets{ip}(3);
    param_gen(4,ip) = param_sets{ip}(4);
    
    % found parameters
    param_fit(1,ip) = mean(cell2mat(epsi_fit(ip,:)));
    param_fit(2,ip) = mean(cell2mat(zeta_fit(ip,:)));
    param_fit(3,ip) = mean(cell2mat(kini_fit(ip,:)));
    param_fit(4,ip) = mean(cell2mat(kinf_fit(ip,:)));
end

%% Compare single parameters (generative-recovered)


%still needs to be adapted for the delta parameter instead of the epsi

figure;
legtxt = {'epsi' 'zeta' 'kini' 'kinf'};
for ip = 1:size(param_gen,1)
    % organize
    par_vals = unique(param_gen(ip,:)); % find all unique parameter values for comparison
    vals_fit = [];
    for par_val = par_vals
        % exclude other parameter fits for special case: epsi == 1 since it
        % overrides the effect of all other parameters
        if ip == 1 && par_val >=.98
            ind_excl = param_gen(ip,:) == par_val;
        end
        ind_par_val = param_gen(ip,:) == par_val;
        if ip ~= 1
            vals_fit = cat(1,vals_fit,[par_val mean(param_fit(ip,ind_par_val&~ind_excl)) std(param_fit(ip,ind_par_val&~ind_excl))/sqrt(ns)]);
        else
            vals_fit = cat(1,vals_fit,[par_val mean(param_fit(ip,ind_par_val)) std(param_fit(ip,ind_par_val))/sqrt(ns)]);
        end
    end
    % plot
    hold on;
    errorbar(vals_fit(:,1),vals_fit(:,2),vals_fit(:,3),'o','LineWidth',2,'CapSize',0,'HandleVisibility','off');
    set(gca,'ColorOrderIndex',ip);
    scatter(vals_fit(:,1),vals_fit(:,2),50,'filled');
end
plot([0 1],[0 1],'k','LineStyle','--'); % reference line
legend(legtxt,'Location','southeast');
title(sprintf('Parameter recovery\nNumber of simulated agents: %d',ns));

%% Check recovery (2 parameters)

% Choose 2 parameters to compare 
% 1/delta, 2/zeta, 3/kini, 4/kinf
param_str = {'delta','zeta','kini','kinf'};
% compare learning noise parameter to epsilon-bias
% organize
i_gen = 1;
i_rec = 2;
par_vals = unique(param_gen(i_gen,:)); % find all unique parameter values for comparison
vals_fit = [];
for par_val = par_vals
    ind_par_val = param_gen(i_gen,:) == par_val;
    % log epsi in x; zeta in y 
    vals_fit = cat(1,vals_fit,[par_val mean(param_fit(i_rec,ind_par_val)) std(param_fit(i_rec,ind_par_val))/sqrt(ns)]);
end
% plot
hold on;
f = errorbar(vals_fit(:,1),vals_fit(:,2),vals_fit(:,3),'o','LineWidth',2,'CapSize',0,'HandleVisibility','off');
set(gca,'ColorOrderIndex',1);
scatter(vals_fit(:,1),vals_fit(:,2),50,'filled');
ylim([0 1])
xlabel(sprintf('Generative parameter: %s',param_str{i_gen}));
ylabel(sprintf('Recovered parameter: %s',param_str{i_rec}));

%% Local functions
function [xt] = resample(m,s,ssel,r)
% 1/ resample (x1-x2)
md = m(1,:)-m(2,:);
sd = sqrt(sum(s.^2,1));
td = tnormrnd(md,sqrt(sd.^2+ssel.^2),r); 
xd = normrnd( ...
    (ssel.^2.*md+sd.^2.*td)./(ssel.^2+sd.^2), ...
    sqrt(ssel.^2.*sd.^2./(ssel.^2+sd.^2)));
% 2/ resample x1 from (x1-x2)
ax = s(1,:).^2./sd.^2;
mx = m(1,:)-ax.*md;
sx = sqrt(s(1,:).^2-ax.^2.*sd.^2);
x1 = ax.*xd+normrnd(mx,sx);
% 3/ return x1 and x2 = x1-(x1-x2)
xt = cat(1,x1,x1-xd);
end

function [x] = tnormrnd(m,s,d)
% sample from truncated normal distribution
for id = 1:numel(d)
    if d(id) == 1
        if m(id) >= 0 
            x(id) = +rpnormv(+m(id),s(id));
        else
            x(id) = +rpnormv(-m(id),s(id));
        end
    else
        if m(id) >= 0
            x(id) = -rpnormv(m(id),s(id));
        else
            x(id) = -rpnormv(-m(id),s(id));
        end
    end
end
end