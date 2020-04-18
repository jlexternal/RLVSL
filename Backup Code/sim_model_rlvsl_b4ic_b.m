%function [out] = sim_model_rlvsl(target_cfg, model_cfg, nsims, modeltype)

% Goal:     Create a function that simulates model data for the RLvSL experiment based on
% model chosen.

% Inputs:   Target configuration for experiment (e.g. # of blocks, block order, # of trials,
% generative mean stdev, etc.)

% Output:   Behavior (i.e. choices)
nsims = 10;

%% Script form for testing before turning into function

% initialize if testing in script form, otherwise comment out
sim_struct  = struct;
sim         = struct;
cfg_model   = struct;

% Model parameters
zeta    = 0.5;  % Learning noise for value learning in KF
beta    = 1;    % Softmax inverse temperature
decay   = 0.2;  % Memory decay rate; element of [0,1] : 0 meaning full memory, 1 meaning full amnesia
kgasy   = 0.0;  % Kalman gain asymptote parameter (multiplicative factor on the sampling variance) 

ns      = nsims;   % Number of model simulations

% Store model parameter values in cfg_model structure
cfg_model.zeta  = zeta;
cfg_model.beta  = beta;
cfg_model.decay = decay;
cfg_model.kgasy = kgasy;

sim_struct.sim.cfg_model = cfg_model;

% Generative parameters of experiment
expe = gen_expe_rlvsl(randi(100)); % generate experimental structure

ntrain  = expe(1).cfg.ntrain;   % number of training blocks
nt      = expe(1).cfg.ntrls;    % number of trials per block
mgen    = expe(1).cfg.mgen;     % mean of generative distribution of expe
sgen    = expe(1).cfg.sgen;     % stdev of generative distribution of expe
nb      = expe(1).cfg.nbout;    % number of true blocks (no training)
nc      = 3;                    % number of different conditions (hard coded)

expe(ntrain+1).cfg = expe(1).cfg;   % get rid of training blocks
expe = expe(4:end);

sim_struct.expe = expe;              % store experimental structure in model_sim structure

% Target parameters for experiment 
mtgt    = 10;               % target mean of distribution
stgt    = 15;               % target std of distribution
mmin    = -mtgt-49;         % min value of outcomes
mmax    = 100-mtgt-51;      % max value of outcomes
a       = stgt/sgen;        % slope of linear transformation aX+b
b       = mtgt - a*mgen;    % intercept of linear transf. aX+b

% Kalman Filter variables
mt = nan(nb,nt,ns);     % posterior mean
vt = nan(nb,nt,ns);     % posterior variance
kt = zeros(nb,nt,ns);   % kalman gain
vs = stgt^2;            % sampling variance
vd = kgasy*vs;          % (perceived) process noise as multiplicative factor of vs

% Structure learning variables
ab = nan(nb,2,ns,nc);   % (ib, alpha/beta, is) parameters of Beta distribution

% Value structures
p_rl = zeros(nb,nt,ns); % p(shape A) from RL
p_sl = zeros(nb,ns);    % p(shape A) from SL
s  = zeros(nb,1,ns);    % log odds from SL
q  = zeros(nb,nt,ns);   % log odds from RL
mp = nan(nb,1,ns);      % subjective probability of shape A being right
dv = nan(nb,nt,ns);     % decision variable
oc = nan(nb,nt,ns);     % outcomes

ib_c = zeros(1,3);       % index of condition blocks

% Simulate experiment
for ib = 1:nb
    
    shapeset    = expe(ib).shape;
    ctype       = expe(ib).type;
    
    switch ctype
        case 'rnd' % random across successive blocks
            ic = 3;
        case 'alt' % always alternating
            ic = 2;
        case 'rep' % always the same
            ic = 1;
    end
    ib_c(ic) = ib_c(ic) + 1;
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%%%%% CODE ALTERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % indexing of the ab array needs to be based on the ith iteration of a block of a
    %   certain condition, running into NaN errors.
    % need to change the way the code handles the indices COMPLETELY
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    oc_temp = round((expe(ib).blck.*a+b)).*ones(1,nt,ns);
    oc_temp(oc_temp>mmax) = mmax;
    oc_temp(oc_temp<mmin) = mmin;
    oc(ib,:,:) = oc_temp;
    
    if ib_c == 1 % first block
        mp(ib,1,:)          = 0.5;      % probability of 1st choice (i.e. shape A)
        mt(ib,1,:)          = 0;        % mean tracked quantity
        vt(ib,1,:)          = 1e2;      % flat RL prior
        ab(ib,:,:,ic) = ones(1,2,ns,1); % flat SL prior (represented by a=b=1)
        p_sl(ib,:)          = 0.5;      % reflects SL parameters above, without having to do calculation
    else
        if ib_c == 2 % second block
            mp(ib,1,:) = 0.5;
            mt(ib,1,:) = 0; % assume no prior bias on tracked quantity
            ab(ib,:,:,ic) = ones(1,2,ns,1); % flat SL prior
            p_sl(ib,:) = 0.5; % reflects SL parameters above, without having to do calculation
            
        else % consider switch/stay prior from Beta distribution
            ab_temp = nan(1,2,ns);
            
            % sign of ib-1 to ib-2 determines switch or stay
            ab_temp(1,1,:) = double(bsxfun(@eq,sign(mt(ib-2,end,:)),sign(mt(ib-1,end,:)))); % stays
            ab_temp(1,2,:) = double(~bsxfun(@eq,sign(mt(ib-2,end,:)),sign(mt(ib-1,end,:)))); % switches
            switchvar               = ab_temp(1,2,:);   % switches are 1
            switchvar(switchvar==0) = -1;               % stays are -1
            ab(ib,1,:,ic)  = ab(ib-1,1,:,ic) + ab_temp(1,1,:).*(1-decay); % alpha++ -decay for stay
            ab(ib,2,:,ic)  = ab(ib-1,2,:,ic) + ab_temp(1,2,:).*(1-decay); % beta++  -decay for switch
            
            % update p_sl where the stay/switch represents shape A
            signswitch = sign(mt(ib-1,end,:)).*switchvar; % sign of last choice * switchvar
            for is = 1:ns
                if sign(mt(ib-1,end,is)) == 1
                    p_sl(ib,is) = ab(ib,1,is,ic)./(ab(ib,1,is,ic)+ab(ib,2,is,ic));
                else
                    p_sl(ib,is) = 1-((ab(ib,1,is,ic))./(ab(ib,1,is,ic)+ab(ib,2,is,ic)));
                end
            end
            
            % update log odds; s-SL, q-RL
            s(ib,1,:) = log(p_sl(ib,:))-log(1-p_sl(ib,:));
            q(ib,1,:) = 0; % log(.5/.5) = 0;
            
            mp(ib,1,:) = 1./(1+exp(-(q(ib,1,:)+s(ib,1,:))));  % probability of shape A on 1st choice
            
            mt(ib,1,:) = 0; % assume no prior bias on tracked quantity
        end
        vt(ib,1,:) = 1e2;% the variance however is not infinite as with the 1st block
    end
    
    % Softmax choice on trial 1
    w_dv = 1./(1+exp(-beta*(mp(ib,1,:)-(1-mp(ib,1,:)))));
    for is=1:ns
        dv(ib,1,is) = datasample(shapeset,1,'Weights',[w_dv(is) 1-w_dv(is)]);
    end
    
    % going through the trials
    for it = 2:nt+1
        
        kt(ib,it-1,:)   = vt(ib,it-1,:)./(vt(ib,it-1,:)+vs);    % Kalman gain update
        mt(ib,it,:)     = mt(ib,it-1,:)+(oc(ib,it-1,:)-mt(ib,it-1,:)).*kt(ib,it-1,:).*(1+randn(1,1,ns)*zeta); % Mean estimate
        vt(ib,it,:)     = (1-kt(ib,it-1,:)).*vt(ib,it-1,:);     % Covariance noise update
        
        p_rl(ib,it,:)   = 1-normcdf(0,mt(ib,it,:),sqrt(vt(ib,it,:)));   % Update contribution from RL
        q(ib,it,:)      = log(p_rl(ib,it,:)) - log(1-p_rl(ib,it,:));    % Convert to logit
        mp(ib,it,:)     = (1+exp(-(q(ib,it,:)+s(ib,1,:)))).^-1;         % Estimate probability of the shape 
        
        % Softmax choice
        w_dv = 1./(1+exp(-beta*(mp(ib,it,:)-(1-mp(ib,it,:)))));
        for is=1:ns
            dv(ib,it,is) = datasample(shapeset,1,'Weights',[w_dv(is) 1-w_dv(is)]);
        end
        
        % Extrapolate covariance for next step
        vt(ib,it,:)     = vt(ib,it,:) + vd; % covariance extrapolation (process noise included i.e. vd)
    end
end

% Store all tracked variables in model_sim structure
sim_struct.sim.q = q;
sim_struct.sim.s = s;
sim_struct.sim.ab = ab;
sim_struct.sim.dv = dv;
sim_struct.sim.kt = kt;
sim_struct.sim.mp = mp;
sim_struct.sim.mt = mt;
sim_struct.sim.vt = vt;
sim_struct.sim.p_rl = p_rl;
sim_struct.sim.p_sl = p_sl;

%end