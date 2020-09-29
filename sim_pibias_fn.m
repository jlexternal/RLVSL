function sim_out = sim_pibias_fn(cfg)
% sim_pibias
%
% Function: Outputs simulation of the trial/block-bias RL model for experiment RLVSL
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
    lscheme     = cfg.lscheme;      % 'ind'-independent action values;  'sym'-symmetric action values
    nscheme     = cfg.nscheme;      % 'rpe'-noise scaling w/ RPE;       'upd'-noise scaling w/ value update
    bscheme     = cfg.bscheme;      % 'trl'-trialwise structure bias    'blk'-blockwise structure bias
    % Model parameters
    ns      = cfg.ns;       % Number of simulated agents to generate per given parameter
    pi      = cfg.pi;       % Probability of the block governed by structure learning (repetition)
    alpha   = cfg.alpha;    % Constant learning rate
    zeta    = cfg.zeta;     % Learning noise scaling parameter
    ksi     = cfg.ksi;      % Learning noise constant parameter
    theta   = cfg.theta;    % Softmax temperature
    % Simulation settings
    if isfield(cfg,'rewcomp')
        rew = cfg.rewcomp;      % reward scheme as input
    else
        rew = [];
    end
    sameexpe = cfg.sameexpe;    % true if all sims see the same exact reward scheme

% softmax spread approximation
ssel = pi/sqrt(6)*theta;

% Generate experiment (reward scheme)
cfg_gb = struct; cfg_gb.ntrls = nt; cfg_gb.mgen = ms; cfg_gb.sgen = sqrt(vs); cfg_gb.nbout = nb;

% generate reward structures if one is not provided
if ~isfield(cfg,'rewcomp')
    rew = cat(3,rew,round(gen_blck_rlvsl(cfg_gb),2));
    if sameexpe
        rew = cat(3,rew,repmat(rew(:,:,1),[1 1 ns-1]));
    else
        for isim = 1:ns-1
            rew = cat(3,rew,round(gen_blck_rlvsl(cfg_gb),2));
        end
    end
end

% track variables
qt = nan(nb,nt,2,ns);   % q-values (chosen/unchosen)
pt = nan(nb,nt,ns);     % response probability
rt = nan(nb,nt,ns);     % actual responses
st = nan(nb,nt,2,ns);   % current-trial filtering noise

% pre-determine structure blocks 
if strcmpi(bscheme,'blk')
    isl_b = rand(nb,ns) < pi;
    isl_b = repmat(reshape(isl_b,[nb 1 ns]),[1 nt 1]);
end

for ib = 1:nb
    for it = 1:nt
        if it == 1
            qt(ib,1,:,:) = .5;
            rt(ib,1,:) = randi(2,1,ns);
            pt(ib,1,:) = .5;
            
            if strcmpi(bscheme,'trl')
                % 1st response from trialwise probablistic structure 
                isl_t = rand(1,ns) < pi;
                % Structure-utilising agents
                if nnz(isl_t) > 0
                    pt(ib,1,isl_t) = 1;
                    rt(ib,1,isl_t) = 1;
                end
            end
            continue
        end
        
        % calculate prediction error
        pe = nan(1,1,2,ns);
        pe(:,:,1,:) = reshape(rew(ib,it-1,:),[1 1 1 ns])-qt(ib,it-1,1,:); 
        pe(:,:,2,:) = 1-reshape(rew(ib,it-1,:),[1 1 1 ns])-qt(ib,it-1,2,:);
        
        % learning noise contribution
        switch nscheme 
            case 'rpe'
                st = sqrt(zeta^2*pe.^2+ksi^2);
            case 'upd'
                st = sqrt(zeta^2*(alpha*pe).^2+ksi^2); 
        end
        st = reshape(st,[1 1 2 ns]);
        
        % update q-values
        ind1 = rt(ib,it-1,:) == 1;  % index of sims choosing option 1
        ind2 = ~ind1;               % index of sims choosing option 2
        % 1/ chosen option
        qt(ib,it,1,ind1) = normrnd(qt(ib,it-1,1,ind1) + alpha*pe(:,:,1,ind1),st(:,:,1,ind1));
        qt(ib,it,2,ind2) = normrnd(qt(ib,it-1,2,ind2) + alpha*pe(:,:,2,ind2),st(:,:,2,ind2));
        % 2/ unchosen option
        switch lscheme
            case 'ind'
                qt(ib,it,1,~ind1) = qt(ib,it-1,1,~ind1);
                qt(ib,it,2,~ind2) = qt(ib,it-1,2,~ind2);
                st(:,:,1,~ind1) = ksi;
                st(:,:,2,~ind2) = ksi;
            case 'sym'
                qt(ib,it,1,~ind1) = normrnd(qt(ib,it-1,1,~ind1) + alpha*pe(:,:,1,~ind1),st(:,:,1,~ind1));
                qt(ib,it,2,~ind2) = normrnd(qt(ib,it-1,2,~ind2) + alpha*pe(:,:,2,~ind2),st(:,:,2,~ind2));
        end
        
        % decision variable stats
        qd = reshape(qt(ib,it,1,:)-qt(ib,it,2,:),[1,ns]);
        sd = reshape(sqrt(sum(st.^2,3)+ssel^2),[1 ns]);
        
        % probability of choosing option 1
        pt(ib,it,:) = 1-normcdf(0,qd,sd);
        
        % extract choice from probabilities
        rt(ib,it,:) = pt(ib,it,:) > rand(1,1,ns);
        
        if strcmpi(bscheme,'trl')
            % sample trial types (based on pi param)
            isl_t = rand(1,ns) < pi;
            % Structure-utilising agents
            if nnz(isl_t) > 0
                pt(ib,it,isl_t) = 1;
                rt(ib,it,isl_t) = 1;
            end
        end
        
        rt(rt==0) = 2;
    end
end
% apply structure bias (overrides RL from loop above)
if strcmpi(bscheme,'blk')
    pt(isl_b==1) = 1;
    rt(isl_b==1) = 1;
end

% output simulation data
sim_out = struct;
sim_out.pi       = pi;
sim_out.alpha    = alpha;
sim_out.zeta     = zeta;
sim_out.ksi      = ksi;
sim_out.theta    = theta;
sim_out.ns       = ns;
sim_out.ms       = ms;
sim_out.vs       = vs;
sim_out.rew      = rew;
sim_out.lscheme  = lscheme;
sim_out.nscheme  = nscheme;
sim_out.bscheme  = bscheme; 
sim_out.pt       = pt;
sim_out.qt       = qt;
sim_out.resp     = rt;

% plot simulation data for DEBUG purposes (set to true)
if false
    rt(rt==2)=0;
    figure(1);
    hold on;
    shadedErrorBar(1:nt,mean(mean(rt,3),1),std(mean(rt,3)),'lineprops',{'LineWidth',2},'patchSaturation',.1);
    ylim([0 1]);
    yline(.5,'--','HandleVisibility','off');
    xticks([1:4]*4);
    xlabel('trial number');
    ylabel('proportion correct');
    title(sprintf('Params: pi:%0.2f, alpha: %0.2f, zeta: %0.2f, theta: %0.2f, \n nsims:%d',pi,alpha,zeta,theta,ns));
    ylim([.4 1]);
end

end