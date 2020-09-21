% delta_parameterization_thombias
%
% Find upper bound value of the delta parameter for the Thompson-bias model (RLVSL)
%
% Jun Seok Lee <jlexternal@gmail.com>

clear all;
addpath('../');
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
lscheme = 'sym';    % 'ind'-independent action values;  'sym'-symmetric action values
nscheme = 'upd';    % 'rpe'-noise scaling w/ RPE;       'upd'-noise scaling w/ value update

% Model parameters
ns      = 27;       % Number of simulated agents to generate per given parameter
ksi     = 0.0+eps;  % Learning noise constant

% Simulation settings
sameexpe = true;   % true if all sims see the same reward scheme

sim_struct = struct;

% Run simulation
% Organize parameter values into sets for simulation

% fixed parameters
zeta   = 0.3; 
kini   = 0.9;
kinf   = 0.1;
theta  = 0.0;

% 0.3 chosen as base value of delta for comparison as the strength of the structure
% is enough to produce no change in behavior upon changing the values of the other
% parameters
delta_0 = .235; 
isfirstrun = true;

finding = true;
while finding
    if isfirstrun
        delta = delta_0;
    else
        delta = delta_1;
    end
    
    ssel = pi/sqrt(6)*theta;

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

    for ib = 1:nb
        for it = 1:nt
            if it == 1
                % initialize KF means and variances
                mt(ib,it,:,:)  = .5;
                % initialize posterior variance based on initial kalman gain parameter
                vt(ib,it,:,:) = kini/(1-kini)*vs;
                % first trial response probability
                pd = normrnd(delta,r_sd,ns,1);
                pt(ib,it,:) = reshape(pd,size(pt(ib,it,:)));
                % first trial response % 1/correct, 2/incorrect
                rt(ib,it,:) = double(pt(ib,it,:)>=0); % argmax choice
                rt(rt~=1) = 2;
                continue;
            end

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
                switch nscheme
                    case 'rpe' % RPE-scaled learning noise
                        st(ib,it,c,ind_c) = sqrt(zeta^2*((rew_seen-mt(ib,it-1,c,ind_c)).^2+ksi^2)); 
                    case 'upd' % update-value-scaled learning noise
                        st(ib,it,c,ind_c) = sqrt(zeta^2*reshape(kt(c,ind_c),size(rew_seen)).*((rew_seen-mt(ib,it-1,c,ind_c)).^2+ksi^2)); 
                end
                % 2/unchosen option
                mt(ib,it,u,ind_c) = mt(ib,it-1,u,ind_c) + reshape(kt(u,ind_c),size(rew_unseen)).*(rew_unseen-mt(ib,it-1,u,ind_c));
                vt(ib,it,u,ind_c) = (1-reshape(kt(u,ind_c),size(rew_unseen))).*vt(ib,it-1,u,ind_c);
                switch nscheme
                    case 'rpe' % RPE-scaled learning noise
                        st(ib,it,u,ind_c) = sqrt(zeta^2*((rew_unseen-mt(ib,it-1,u,ind_c)).^2+ksi^2));
                    case 'upd' % update-value-scaled learning noise
                        st(ib,it,u,ind_c) = sqrt(zeta^2*reshape(kt(u,ind_c),size(rew_unseen)).*((rew_unseen-mt(ib,it-1,u,ind_c)).^2+ksi^2));
                end
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

        end
    end

    rt(rt==2) = eps;
    p = sum(rt,3)./ns;

    % plot simulation data
    if false
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
    
    if isfirstrun
        p1 = p(:);
        delta_1 = .2;
        isfirstrun = false;
        continue
    else
        p2 = p(:);
        kld = kldiv(p1,p2);
        % Look for the lowest value of delta_1 where the KL divergence with delta_0 is 0
        fprintf('delta = %0.3f; kldiv = %0.6f\n',delta_1,kld);
        delta_1 = delta_1+.001;
    end
    
    
    if delta_1 >= delta_0 
        finding = false;
    end
        
end

% local functions

function out = kldiv(p1,p2)
    out = sum(p1.*(log(p1)-log(p2)));
end