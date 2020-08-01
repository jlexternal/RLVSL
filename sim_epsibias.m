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
% Model parameters
ns      = 5;       % Number of simulated agents to generate per given parameter
kini    = 0.8-eps;  % Initial Kalman gain
kinf    = 0.3+eps;  % Asymptotic Kalman gain
zeta    = 0.3+eps;  % Learning noise scale
ksi     = 0.0+eps;  % Learning noise constant
epsis   = 0.2;      % Blind structure choice 0: all RL, 1: all SL

params_gen = struct;
params_gen.kini = kini;
params_gen.kinf = kinf;
params_gen.zeta = zeta;
params_gen.ksi  = ksi;

sbias_cor = true;   % bias toward the correct structure
sbias_ini = true;   % initial biased means

% Simulation settings
sameexpe = false;   % true if all sims see the same reward scheme
nexp     = 10;      % number of different reward schemes to try per given parameter set

sim_struct = struct;

%% Run simulation

% Organize parameter sets for simulation
epsis = linspace(0,.99,4);
zetas = [0:.1:.4]+eps;
kinis = .9;%.5:.1:1;
kinfs = .2;%0:.1:.4;
param_sets = {};
p_ctr = 0;
for epsi = epsis
    for zeta = zetas
        for kini = kinis
            for kinf = kinfs
                p_ctr = p_ctr + 1;
                param_sets{p_ctr} = [epsi,zeta,kini,kinf];
            end
        end
    end
end

out_ctr = 0;
for ip = 1:numel(param_sets)
    epsi = param_sets{ip}(1);
    zeta = param_sets{ip}(2);
    kini = param_sets{ip}(3);
    kinf = param_sets{ip}(4);
    
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
                rew_c(ib,it-1,ind_c) = rew_seen; % output for recovery
                
                rew_seen    = reshape(rew_seen,size(mt(ib,it-1,c,ind_c)));
                rew_unseen  = reshape(rew_unseen,size(mt(ib,it-1,c,ind_c)));
                mt(ib,it,c,ind_c) = mt(ib,it-1,c,ind_c) + reshape(kt(c,ind_c),size(rew_seen)).*(rew_seen-mt(ib,it-1,c,ind_c));
                vt(ib,it,c,ind_c) = (1-reshape(kt(c,ind_c),size(rew_seen))).*vt(ib,it-1,c,ind_c);
                st(ib,it,c,ind_c) = sqrt(zeta^2*((rew_seen-mt(ib,it-1,c,ind_c)).^2+ksi^2)); % RPE-scaled learning noise
                
                mt(ib,it,u,ind_c) = mt(ib,it-1,u,ind_c) + reshape(kt(u,ind_c),size(rew_unseen)).*(rew_unseen-mt(ib,it-1,u,ind_c));
                vt(ib,it,u,ind_c) = (1-reshape(kt(u,ind_c),size(rew_unseen))).*vt(ib,it-1,u,ind_c);
                st(ib,it,u,ind_c) = sqrt(zeta^2*((rew_unseen-mt(ib,it-1,u,ind_c)).^2+ksi^2));
            end
            % variance extrapolation + diffusion process 
            vt(ib,it,:,:)  = vt(ib,it,:,:)+fv(kinf); % covariance noise update    
            % selection noise
            ssel = 0;
            % decision variable stats
            md = reshape(mt(ib,it,1,:)-mt(ib,it,2,:),[1 ns]);
            sd = reshape(sqrt(sum(st(ib,it,:,:).^2,3)+ssel^2),[1 ns]);
            % sample trial types (based on epsi param)
            isl = rand(1,ns) < epsi;
            irl = ~isl;
             % Structure-utilising agents
            if nnz(isl) > 0
                pt(ib,it,isl) = rb(isl) == 1;
            end
            % RL-utilising agents
            if nnz(irl) > 0
                pt(ib,it,irl) = 1-normcdf(0,md(irl),sd(irl));
            end
            % extract choice from probabilities
            rt(ib,it,:) = round(pt(ib,it,:));
            rt(rt==0) = 2;
            % resampling w/o conditioning on response
            mt(ib,it,:,isl) = normrnd(mt(ib,it,:,isl),st(ib,it,:,isl));
            % resampling conditioning on response
            if nnz(irl) > 0
                mt(ib,it,:,irl) = resample(reshape(mt(ib,it,:,irl),[2,nnz(irl)]),...
                                           reshape(st(ib,it,:,irl),[2,nnz(irl)]),...
                                           ssel,reshape(rt(ib,it,irl),[1 numel(irl(irl==1))]));
            end
        end
    end
    
    % output simulation data
    sim_struct(out_ctr).epsi = epsi;
    sim_struct(out_ctr).zeta = zeta;
    sim_struct(out_ctr).kini = kini;
    sim_struct(out_ctr).kinf = kinf;
    sim_struct(out_ctr).ksi  = ksi;
    sim_struct(out_ctr).vs   = vs;
    sim_struct(out_ctr).resp = rt;
    sim_struct(out_ctr).rew_seen = rew_c;
    
    % plot simulation data
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
        title(sprintf('Params: kini:%0.2f, kinf: %0.2f, zeta: %0.2f, epsi: %0.2f\n nsims:%d',kini,kinf,zeta,epsi,ns));
        ylim([.4 1]);
    end
end
clearvars -except param_sets sim_struct ns

%% Parameter recovery (fit model to simulated data)

if ~bsxfun(@eq,numel(sim_struct),numel(param_sets))
    error('Number of parameter sets does not match the number of simulation outputs!');
end

for ip = 1:numel(param_sets)
    epsi = param_sets{ip}(1);
    zeta = param_sets{ip}(2);
    kini = param_sets{ip}(3);
    kinf = param_sets{ip}(4);
    
    for isim = 1:ns
        cfg = [];
        cfg.resp = sim_struct(ip).resp(:,:,isim);
        cfg.rt = sim_struct(ip).rew_seen(:,:,isim);
        cfg.vs = sim_struct(ip).vs;
        cfg.nsmp = 1e3;
        cfg.lstruct = 'sym'; % assume symmetric action values
        cfg.verbose = true; % plot fitting information
        cfg.ksi = 0; % assume no constant term in learning noise

        out_fit{ip,isim} = fit_noisyKF_epsibias(cfg); % fit the model to data
        
        epsi_fit{ip,isim} = out_fit{ip,isim}.epsi;
        zeta_fit{ip,isim} = out_fit{ip,isim}.zeta;
        kini_fit{ip,isim} = out_fit{ip,isim}.kini;
        kinf_fit{ip,isim} = out_fit{ip,isim}.kinf;
    end
end
savename = ['fit_struct_epsibias_' datestr(now,'ddmmyyyy')];
save(savename,'out_fit','param_sets','sim_struct','epsi_fit','zeta_fit','kini_fit','kinf_fit');
%% Check recovery (single parameter)

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

%% Single parameter generative-recovered comparisons
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
    f = errorbar(vals_fit(:,1),vals_fit(:,2),vals_fit(:,3),'o','LineWidth',2,'CapSize',0,'HandleVisibility','off');
    set(gca,'ColorOrderIndex',ip);
    scatter(vals_fit(:,1),vals_fit(:,2),50,'filled')
end
plot([0 1],[0 1],'k','LineStyle','--') % reference line
legend(legtxt,'Location','southeast')
title(sprintf('Parameter recovery\nNumber of simulated agents: %d',ns))

%% Check recovery (2 parameters)

% Choose 2 parameters to compare 
% 1/epsi, 2/zeta, 3/kini, 4/kinf
param_str = {'epsi','zeta','kini','kinf'};
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
scatter(vals_fit(:,1),vals_fit(:,2),50,'filled')
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
        x(id) = +rpnormv(+m(id),s(id));
    else
        x(id) = -rpnormv(-m(id),s(id));
    end
end
end