function [out] = gen_sim_model1_rlvsl(target_cfg, model_cfg)
% sim_model_rlvsl
%
% Simulates data coming from model 1 
% (4 parameter model with structure learning as updates of a Beta distribution)
%
% Goal:     Create a function that simulates model data for the RLvSL experiment based on
% model chosen.
%
% Inputs:   Target configuration for experiment (e.g. # of blocks, block order, # of trials,
% generative mean stdev, etc.)
%           model_cfg:  structure containing parameters pertaining to the generative model
%                       .nsims 
%                       .zeta
%                       .beta
%                       .gamma
%                       .kappa
%           target_cfg: structure containing parameters of target distributions
%                       .mtgt
%                       .stgt
%
% Output:   Behavior (i.e. choices)
% 
% - Jun Seok Lee (December 2019) jlexternal@gmail.com 

% Initialize random number generator
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

mtgt    = target_cfg.mtgt;               % Mean of target distribution - comes from input cfg
stgt    = target_cfg.stgt;               % Std of target distribution  - comes from input cfg

cfg_tgt.mtgt = mtgt;
cfg_tgt.stgt = stgt;
sim_struct.sim.cfg_tgt = cfg_tgt;

% Model parameters
ns      = model_cfg.nsims;  % Number of simulations to generate per given parameter
zeta    = model_cfg.zeta;   % Learning noise for value learning in KF
beta    = model_cfg.beta;	% Softmax inverse temperature
gamma   = model_cfg.gamma; 	% Memory decay rate; element of [0,1] : 0 meaning full memory, 1 meaning full amnesia
kappa   = model_cfg.kappa;  % Kalman gain asymptote parameter (multiplicative factor on the sampling variance) 

% Store model parameter values in cfg_model structure
cfg_model.zeta  = zeta;
cfg_model.beta  = beta;
cfg_model.gamma = gamma;
cfg_model.kappa = kappa;

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

sim_struct.expe = expe;         % store experimental structure in model_sim structure

% Target parameters for experiment 
omin    = -mtgt-49;         % min value of outcomes
omax    = 100-mtgt-51;      % max value of outcomes
a       = stgt/sgen;        % slope of linear transformation aX+b
b       = mtgt - a*mgen;    % intercept of linear transf. aX+b

% Kalman Filter variables
mt = nan(nb/3,nt,ns,nc);     % posterior mean
vt = nan(nb/3,nt,ns,nc);     % posterior variance
kt = zeros(nb/3,nt,ns,nc);   % kalman gain
vs = stgt^2;            % sampling variance
vd = kappa*vs;          % (perceived) process noise as multiplicative factor of vs

% Structure learning variables
ab = nan(nb/3,2,ns,nc);   % (ib, alpha/beta, is) parameters of Beta distribution

% Value structures
p_rl = zeros(nb/3,nt,ns,nc); % p(shape A) from RL
p_sl = zeros(nb/3,ns,nc);    % p(shape A) from SL
s  = zeros(nb/3,1,ns,nc);    % log odds from SL
q  = zeros(nb/3,nt,ns,nc);   % log odds from RL
mp = nan(nb/3,1,ns,nc);      % subjective probability of shape A being right
dv = nan(nb/3,nt,ns,nc);     % decision variable
oc = nan(nb/3,nt,ns,nc);     % outcomes

ib_c        = zeros(1,3);    % index of condition blocks
decay_trckr = zeros(nb/3,3); % track blocks of a certain condition for decay
pos_shape   = zeros(3,1);    % tracks the previous good shape for outcome signage
out_mult    = 1;             % multiplier on the outcome for signage based on condition

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
    
    % condition-based indexing of blocks
    ib_c(ic)                 = ib_c(ic) + 1;
    decay_trckr(ib_c(ic),ic) = ib;
    
    % outcome data structure
    if ib < 4
        pos_shape(ic) = shapeset(1);
    end
    
    if ib >= 4
        if shapeset(1) ~= pos_shape(ic)
            out_mult = -1;
        else
            out_mult = 1;
        end
    end
    
    oc_temp = round((out_mult.*expe(ib).blck.*a+b)).*ones(1,nt,ns);
    oc_temp(oc_temp>omax) = omax;
    oc_temp(oc_temp<omin) = omin;
    oc(ib_c(ic),:,:,ic) = oc_temp;
    
    if ib_c(ic) == 1    % first block
        mp(ib_c(ic),1,:,ic)	= 0.5;              % probability of 1st choice (i.e. shape A)
        mt(ib_c(ic),1,:,ic) = 0;                % mean tracked quantity
        vt(ib_c(ic),1,:,ic) = 1e4;              % flat RL prior
        ab(ib_c(ic),:,:,ic) = ones(1,2,ns,1);   % flat SL prior (represented by a=b=1)
        p_sl(ib_c(ic),:,ic) = 0.5;              % reflects SL parameters above, without having to do calculation
    else
        if ib_c(ic) == 2  % second block
            mp(ib_c(ic),1,:,ic) = 0.5;
            mt(ib_c(ic),1,:,ic) = 0;                % assume no prior bias on tracked quantity
            ab(ib_c(ic),:,:,ic) = ones(1,2,ns,1);   % flat SL prior
            p_sl(ib_c(ic),:,ic) = 0.5;              % reflects SL parameters above, without having to do calculation
            
        elseif ib_c(ic)>2 % consider switch/stay prior from Beta distribution
            ab_temp = nan(1,2,ns);
            
            % sign of ib-1 to ib-2 determines switch or stay
            ab_temp(1,1,:) = double(bsxfun(@eq,sign(mt(ib_c(ic)-2,end,:,ic)),sign(mt(ib_c(ic)-1,end,:,ic))));  % stays
            ab_temp(1,2,:) = double(~bsxfun(@eq,sign(mt(ib_c(ic)-2,end,:,ic)),sign(mt(ib_c(ic)-1,end,:,ic)))); % switches
            
            if ic ~= 3
                ab(ib_c(ic),1,:,ic)     = ab(ib_c(ic)-1,1,:,ic) + ab_temp(1,1,:).*(1-gamma); %*(decay_trckr(ib_c(ic),ic)-decay_trckr(ib_c(ic)-1,ic))) % alpha++ -decay for stay
                ab(ib_c(ic),2,:,ic)     = ab(ib_c(ic)-1,2,:,ic) + ab_temp(1,2,:).*(1-gamma); %*(decay_trckr(ib_c(ic),ic)-decay_trckr(ib_c(ic)-1,ic))) % beta++  -decay for switch
            else
                % Random condition does not consider stay/switch 
                ab(ib_c(ic),:,:,ic)     = ab(ib_c(ic)-1,:,:,ic);
            end
            % update p_sl where the stay/switch represents shape A
            for is = 1:ns
                if sign(mt(ib_c(ic)-1,end,is,ic)) == 1
                    p_sl(ib_c(ic),is,ic) = ab(ib_c(ic),1,is,ic)./(ab(ib_c(ic),1,is,ic)+ab(ib_c(ic),2,is,ic));
                else
                    p_sl(ib_c(ic),is,ic) = 1-((ab(ib_c(ic),1,is,ic))./(ab(ib_c(ic),1,is,ic)+ab(ib_c(ic),2,is,ic)));
                end
            end
            
            % update log odds; s-SL, q-RL
            s(ib_c(ic),1,:,ic) = log(p_sl(ib_c(ic),:,ic))-log(1-p_sl(ib_c(ic),:,ic));
            q(ib_c(ic),1,:,ic) = 0; % log(.5/.5) = 0;
            
            mp(ib_c(ic),1,:,ic) = 1./(1+exp(-(q(ib_c(ic),1,:,ic)+s(ib_c(ic),1,:,ic))));  % probability of shape A on 1st choice
            
            mt(ib_c(ic),1,:,ic) = 0; % assume no prior bias on tracked quantity
        end
        vt(ib_c(ic),1,:,ic) = 1e4;
    end
    
    % Softmax choice on trial 1
    w_dv = 1./(1+exp(-beta*(mp(ib_c(ic),1,:,ic)-(1-mp(ib_c(ic),1,:,ic)))));
    
    for is = 1:ns
        dv(ib_c(ic),1,is,ic) = datasample(shapeset,1,'Weights',[w_dv(is) 1-w_dv(is)]);
    end
    
    % going through the trials
    for it = 2:nt
        kt(ib_c(ic),it-1,:,ic)   = vt(ib_c(ic),it-1,:,ic)./(vt(ib_c(ic),it-1,:,ic)+vs);    % Kalman gain update
        mt(ib_c(ic),it,:,ic)     = mt(ib_c(ic),it-1,:,ic)+ ...                             % Mean estimate
                                   (oc(ib_c(ic),it-1,:,ic)-mt(ib_c(ic),it-1,:,ic)).*kt(ib_c(ic),it-1,:,ic).*(1+randn(1,1,ns)*zeta);
        vt(ib_c(ic),it,:,ic)     = (1-kt(ib_c(ic),it-1,:,ic)).*vt(ib_c(ic),it-1,:,ic);     % Covariance noise update
        
        p_rl(ib_c(ic),it,:,ic)   = 1-normcdf(0,mt(ib_c(ic),it,:,ic),sqrt(vt(ib_c(ic),it,:,ic)));   % Update contribution from RL
        q(ib_c(ic),it,:,ic)      = log(p_rl(ib_c(ic),it,:,ic)) - log(1-p_rl(ib_c(ic),it,:,ic));    % Convert to logit
        mp(ib_c(ic),it,:,ic)     = (1+exp(-(q(ib_c(ic),it,:,ic)+s(ib_c(ic),1,:,ic)))).^-1;         % Estimate probability of the shape 
        
        % Softmax choice
        w_dv = 1./(1+exp(-beta*(mp(ib_c(ic),it,:,ic)-(1-mp(ib_c(ic),it,:,ic)))));
        for is=1:ns
            dv(ib_c(ic),it,is,ic) = datasample(shapeset,1,'Weights',[w_dv(is) 1-w_dv(is)]);
        end
        
        % Extrapolate covariance for next step
        vt(ib_c(ic),it,:,ic)     = vt(ib_c(ic),it,:,ic) + vd; % covariance extrapolation (process noise included i.e. vd)
    end
    
    
end
% Store all tracked variables in sim_struct structure
sim_struct.sim.q    = q;
sim_struct.sim.s    = s;
sim_struct.sim.ab   = ab;
sim_struct.sim.dv   = dv;
sim_struct.sim.kt   = kt;
sim_struct.sim.mp   = mp;
sim_struct.sim.mt   = mt;
sim_struct.sim.vt   = vt;
sim_struct.sim.p_rl = p_rl;
sim_struct.sim.p_sl = p_sl;

%{
 Store all tracked variables in sim_struct structure
sim_struct.sim.q(ib_c(ic),:,:,ic) = q(ib_c(ic),:,:,ic);
sim_struct.sim.s(ib_c(ic),:,:,ic) = s(ib_c(ic),:,:,ic);
sim_struct.sim.ab(ib_c(ic),:,:,ic) = ab(ib_c(ic),:,:,ic);
sim_struct.sim.dv(ib_c(ic),:,:,ic) = dv(ib_c(ic),:,:,ic);
sim_struct.sim.kt(ib_c(ic),:,:,ic) = kt(ib_c(ic),:,:,ic);
sim_struct.sim.mp(ib_c(ic),:,:,ic) = mp(ib_c(ic),:,:,ic);
sim_struct.sim.mt(ib_c(ic),:,:,ic) = mt(ib_c(ic),:,:,ic);
sim_struct.sim.vt(ib_c(ic),:,:,ic) = vt(ib_c(ic),:,:,ic);
sim_struct.sim.p_rl(ib_c(ic),:,:,ic) = p_rl(ib_c(ic),:,:,ic);
sim_struct.sim.p_sl(ib_c(ic),:,ic) = p_sl(ib_c(ic),:,ic);
%}


out.sim_struct = sim_struct;
end