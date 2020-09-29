function sim_out = sim_epsibias_fn(cfg)
% sim_epsibias_fn
%
% Simulates the epsilon-greedy bias model for experiment RLVSL
% (See below for required configuration values)
%
% Jun Seok Lee <jlexternal@gmail.com>

% Required configuration values
    % Experimental parameters
    nb = cfg.nb; % number of blocks
    nt = cfg.nt; % number of trials
    % Generative parameters of winning distribution with FNR of 25%
    ms = cfg.ms; % sampling mean
    vs = cfg.vs; % sampling variance
    % Assumptions of the model
    sbias_cor   = cfg.sbias_cor;    % 1st-choice bias toward the correct structure (set to FALSE to match subject)
    sbias_ini   = cfg.sbias_ini;    % KF means biased toward the correct structure
    cscheme     = cfg.cscheme;      % 'qvs'-softmax;                    'ths'-Thompson sampling
    lscheme     = cfg.lscheme;      % 'ind'-independent action values;  'sym'-symmetric action values
    nscheme     = cfg.nscheme;      % 'rpe'-noise scaling w/ RPE;       'upd'-noise scaling w/ value update
    % Model parameters
    ns      = cfg.ns;   % Number of simulated agents to generate per given parameter
    epsi    = cfg.epsi;
    kini    = cfg.kini;
    kinf    = cfg.kinf;
    zeta    = cfg.zeta;
    ksi     = cfg.ksi;
    theta   = cfg.theta;
    % Simulation settings
    sameexpe = cfg.sameexpe;    % true if all sims see the same reward scheme
%   cfg.compexpe is a matrix of the rewards SET in the experiment (not SEEN)
%   cfg.firstresp is a vector of first responses
    
% reparameterization functions
fk = @(v)1./(1+exp(+0.4486-log2(v/vs)*0.6282)).^0.5057;
fv = @(k)fzero(@(v)fk(v)-min(max(k,0.001),0.999),vs.*2.^[-30,+30]);

if ~isfield(cfg,'firstresp')
    disp('1st responses not provided for comparison! Using random values!');
else
    firstresp   = cfg.firstresp;    % vector of 1st-response for each block
    if numel(cfg.firstresp) ~= nb
        error('Number of first responses does not match the number of blocks!');
    end
    firstresp = reshape(firstresp,[nb,1]); % reshape to column vector
end

sim_out = struct;

if ~ismember(cscheme,{'arg','qvs','ths'})
    error('Undefined or unrecognized choice sampling scheme!');
end

% softmax spread approximation
ssel = pi/sqrt(6)*theta;

% Generate experiment (reward scheme)
cfg_gb = struct; cfg_gb.ntrls = nt; cfg_gb.mgen = ms; cfg_gb.sgen = sqrt(vs); cfg_gb.nbout = nb;
if ~isfield(cfg,'compexpe')
    % generate reward structures (not comparing)
    rew = []; % (nb,nt,ns)
    rew = cat(3,rew,round(gen_blck_rlvsl(cfg_gb),2));
else
    compexpe = cfg.compexpe;    % experimental rewards to compare
    % check for reward structure dimensions
    if ~bsxfun(@eq,size(compexpe),size(ones(nb,nt)))
        error('Dimensions of experiment rewards provided do not match experimental settings!');
    else
        rew = compexpe;
    end
    % if doing a comparing responses, should use the same rew structure as base
    if ~sameexpe
        disp('Comparing experiments; sameexpe is forced to be true!');
        sameexpe = true;
    end
end
if sameexpe
    rew = cat(3,rew,repmat(rew(:,:,1),[1 1 ns-1]));
else
    for isim = 1:ns-1
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
                if ~isfield(cfg,'firstresp')
                    % random 1st choice
                    rb = randi(2,1,ns);
                else
                    % match 1st choices of simulations to the one provided 
                    rb = firstresp(ib).*ones(1,ns);
                end
            end
            % initialize KF means and variances
            if sbias_ini
                % initial tracking mean biased toward generative mean
                ind_rb = rb == 1;
                mt(ib,it,1,ind_rb)  = ms;
                mt(ib,it,1,~ind_rb) = 1-ms;
                mt(ib,it,2,:) = 1-mt(ib,it,1,:);
            else
                % initial tracking mean unbiased
                mt(ib,it,:,:) = 0.5;
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
        
        % update Kalman gain
        kt = reshape(vt(ib,it-1,:,:)./(vt(ib,it-1,:,:)+vs),[2 ns]);
        % update posterior mean & variance
        for io = 1:2
            ind_c = find(rt(ib,it-1,:)==io); % index of sims that chose option 1 or 2
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
            
            % Update tracked mean from KF
            mt(ib,it,c,ind_c) = mt(ib,it-1,c,ind_c) + reshape(kt(c,ind_c),size(rew_seen)).*(rew_seen-mt(ib,it-1,c,ind_c));
            % Calculate learning noise contribution to spread
            switch nscheme
                case 'rpe' % RPE-scaled learning noise
                    st(ib,it,c,ind_c) = sqrt(zeta^2*(rew_seen-mt(ib,it-1,c,ind_c)).^2+ksi^2); 
                case 'upd' % update-value-scaled learning noise
                    st(ib,it,c,ind_c) = sqrt(zeta^2*(reshape(kt(c,ind_c),size(rew_seen)).*(rew_seen-mt(ib,it-1,c,ind_c)).^2+ksi^2)); 
            end
            % Update mean after learning noise contribution
            mt(ib,it,c,ind_c) = normrnd(mt(ib,it,c,ind_c), ...  
                                        st(ib,it,c,ind_c));
            % Update posterior variance
            vt(ib,it,c,ind_c) = (1-reshape(kt(c,ind_c),size(rew_seen))).*vt(ib,it-1,c,ind_c);
            
            % 2/unchosen option
            switch lscheme
                case 'ind'
                    mt(ib,it,u,ind_c) = mt(ib,it-1,u,ind_c);
                    vt(ib,it,u,ind_c) = vt(ib,it-1,u,ind_c);
                case 'sym'
                    mt(ib,it,u,ind_c) = mt(ib,it-1,u,ind_c) + reshape(kt(u,ind_c),size(rew_unseen)).*(rew_unseen-mt(ib,it-1,u,ind_c));
                    switch nscheme
                        case 'rpe'
                            st(ib,it,u,ind_c) = sqrt(zeta^2*((rew_unseen-mt(ib,it-1,u,ind_c)).^2+ksi^2));
                        case 'upd'
                            st(ib,it,u,ind_c) = sqrt(zeta^2*reshape(kt(u,ind_c),size(rew_unseen)).*((rew_unseen-mt(ib,it-1,u,ind_c)).^2+ksi^2));
                    end
                    mt(ib,it,u,ind_c) = normrnd(mt(ib,it,u,ind_c), ...
                                                st(ib,it,u,ind_c));
                    vt(ib,it,u,ind_c) = (1-reshape(kt(u,ind_c),size(rew_unseen))).*vt(ib,it-1,u,ind_c);
            end
        end
        % variance extrapolation/diffusion process 
        vt(ib,it,:,:)  = vt(ib,it,:,:)+fv(kinf); % covariance noise update

        % decision variable stats
        switch cscheme
            case 'qvs'
                md = reshape(mt(ib,it,1,:)-mt(ib,it,2,:),[1 ns]);
                sd = reshape(sqrt(sum(st(ib,it,:,:).^2,3)+ssel^2),[1 ns]);
            case 'ths'
                md = reshape((mt(ib,it,1,:)-mt(ib,it,2,:))./sqrt(sum(vt(ib,it,:,:))),[1 ns]);
                sd = reshape(sqrt(sum(st(ib,it,:,:).^2,3)./sum(vt(ib,it,:,:),3)+ssel^2),[1 ns]);
        end

        % sample trial types (based on epsi param)
        isl = rand(1,ns) < epsi;
        irl = ~isl;
         % Structure-utilising agents
        if nnz(isl) > 0
            rt(ib,it,isl) = rb(isl);
        end
        % RL-utilising agents
        if nnz(irl) > 0
            pt(ib,it,irl) = 1-normcdf(0,md(irl),sd(irl));
            rt(ib,it,irl) = pt(ib,it,irl) > rand(1,1,nnz(irl));
        end
        % extract choice from probabilities
        rt(rt==0) = 2;
    end
end

% output simulation data
sim_out.epsi     = epsi;
sim_out.kini     = kini;
sim_out.kinf     = kinf;
sim_out.zeta     = zeta;
sim_out.ksi      = ksi;
sim_out.theta    = theta;
sim_out.ns       = ns;
sim_out.ms       = ms;
sim_out.vs       = vs;
sim_out.resp     = rt;
sim_out.rew      = rew;
sim_out.rew_seen = rew_c;
sim_out.sbias_cor   = sbias_cor;
sim_out.sbias_ini   = sbias_ini;
sim_out.cscheme     = cscheme;
sim_out.lscheme     = lscheme;
sim_out.nscheme     = nscheme;

% plot simulation data for DEBUG purposes (set to true)
if false
    rt(rt==2)=0;
    figure(1);
    hold on;
    shadedErrorBar(1:nt,mean(mean(rt,3),1),std(mean(rt,3)),'lineprops',{'LineWidth',2+epsi},'patchSaturation',.1);
    ylim([0 1]);
    yline(.5,'--','HandleVisibility','off');
    xticks([1:4]*4);
    xlabel('trial number');
    ylabel('proportion correct');
    title(sprintf('Params: kini:%0.2f, kinf: %0.2f, zeta: %0.2f, theta: %0.2f, epsi: %0.2f\n nsims:%d',kini,kinf,zeta,theta,epsi,ns));
    ylim([.4 1]);
end

end
