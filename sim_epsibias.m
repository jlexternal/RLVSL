% sim_epsibias

% simulate model with epsilon-greedy bias
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
ns      = 20;       % Number of simulated agents to generate per given parameter
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

out_ctr = 0;
for epsi = epsis
    params_gen.epsi = epsi;
    out_ctr = out_ctr + 1;
    % Generate experiment (reward scheme)
    cfg_gb = struct; cfg_gb.ntrls = nt; cfg_gb.mgen = r_mu; cfg_gb.sgen = r_sd; cfg_gb.nbout = nb;
    rew = []; % (nb,nt,ns)
    rew = cat(3,rew,round(gen_blck_rlvsl(cfg_gb),2));
    if sameexpe
        rew = cat(3,rew,repmat(rew(:,:,1),[1 1 ns-1]));
    else
        for is = 1:ns-1
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
    rb = nan(ns,1);           % response bias
    
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
                    mt(ib,it,1,find(rb==1)) = r_mu;
                    mt(ib,it,1,find(rb==2)) = 1-r_mu;
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
                idx_c = find(rt(ib,it-1,:)==io); % index of sims
                c = io;     % chosen option
                u = 3-io;   % unchosen option
                if io == 1
                    rew_seen    = rew(ib,it-1,idx_c);
                    rew_unseen  = 1-rew(ib,it-1,idx_c);
                    if it == nt
                        rew_c(ib,it,idx_c) = rew(ib,it,idx_c);
                    end
                else
                    rew_seen    = 1-rew(ib,it-1,idx_c);
                    rew_unseen  = rew(ib,it-1,idx_c);
                    if it == nt
                        rew_c(ib,it,idx_c) = 1-rew(ib,it,idx_c);
                    end
                end
                rew_c(ib,it-1,idx_c) = rew_seen;
                
                
                rew_seen    = reshape(rew_seen,size(mt(ib,it-1,c,idx_c)));
                rew_unseen  = reshape(rew_unseen,size(mt(ib,it-1,c,idx_c)));
                mt(ib,it,c,idx_c) = mt(ib,it-1,c,idx_c) + reshape(kt(c,idx_c),size(rew_seen)).*(rew_seen-mt(ib,it-1,c,idx_c));
                vt(ib,it,c,idx_c) = (1-reshape(kt(c,idx_c),size(rew_seen))).*vt(ib,it-1,c,idx_c);
                st(ib,it,c,idx_c) = sqrt(zeta^2*((rew_seen-mt(ib,it-1,c,idx_c)).^2+ksi^2)); % RPE-scaled learning noise
                
                mt(ib,it,u,idx_c) = mt(ib,it-1,u,idx_c) + reshape(kt(u,idx_c),size(rew_unseen)).*(rew_unseen-mt(ib,it-1,u,idx_c));
                vt(ib,it,u,idx_c) = (1-reshape(kt(u,idx_c),size(rew_unseen))).*vt(ib,it-1,u,idx_c);
                st(ib,it,u,idx_c) = sqrt(zeta^2*((rew_unseen-mt(ib,it-1,u,idx_c)).^2+ksi^2));
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
    sim_struct(out_ctr).kini = kini;
    sim_struct(out_ctr).kinf = kinf;
    sim_struct(out_ctr).zeta = zeta;
    sim_struct(out_ctr).ksi  = ksi;
    sim_struct(out_ctr).vs   = vs;
    sim_struct(out_ctr).resp = rt;
    sim_struct(out_ctr).rews = rew_c;
    
    rt(rt==2)=0;
end


%% plotting
figure(1);
hold on;
shadedErrorBar(1:nt,mean(mean(rt,3),1),std(mean(rt,3))/sqrt(ns),...
                'lineprops',{'LineWidth',2+epsi},'patchSaturation',.1);
ylim([0 1]);
yline(.5,'--','HandleVisibility','off');
xticks([1:4]*4);
xlabel('trial number');
ylabel('proportion correct');
title(sprintf('Params: kini:%0.2f, kinf: %0.2f, zeta: %0.2f, epsi: %0.2f\n nsims:%d',kini,kinf,zeta,epsi,ns))
%end

ylim([.2 1]);


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
for id = d:numel(d)
    if d(id) == 1
        x(id) = +rpnormv(+m(id),s(id));
    else
        x(id) = -rpnormv(-m(id),s(id));
    end
end
end