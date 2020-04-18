function [out] = gen_sim_model2_rlvsl(target_cfg, model_cfg)
% gen_sim_model1_rlvsl
%
% Simulates data coming from model 2 
% (2 parameter model with structure learning fitted directly over each quarter)
%
% Goal:     Create a function that simulates model data for the RLvSL experiment based on
% model chosen.
%
% Inputs:   Target configuration for experiment (e.g. # of blocks, block order, # of trials,
% generative mean stdev, etc.)
%           model_cfg:  structure containing parameters pertaining to the generative model
%                       .nsims 
%                       .zeta
%                       .strpr 
%                       .alpha
%           target_cfg: structure containing parameters of target distributions
%                       .mtgt
%                       .stgt
%
% Output:   Behavior (i.e. choices)
% 
% Jun Seok Lee  - <jlexternal@gmail.com>
% February 2020

mtgt    = target_cfg.mtgt;               % Mean of target distribution - comes from input cfg
stgt    = target_cfg.stgt;               % Std of target distribution  - comes from input cfg

cfg_tgt.mtgt = mtgt;
cfg_tgt.stgt = stgt;
sim_struct.sim.cfg_tgt = cfg_tgt;

% Model parameters
ns      = model_cfg.nsims;  % Number of simulations to generate per given parameter
zeta    = model_cfg.zeta;   % Learning noise for value learning in KF
alpha   = model_cfg.alpha;  % Kalman gain asymptote parameter (multiplicative factor on the sampling variance) 
strpr   = model_cfg.strpr;  % value for the prior probability of choosing shape A via structure learning for each condition

% Store model parameter values in cfg_model structure
cfg_model.zeta  = zeta;
cfg_model.alpha = alpha;
cfg_model.strpr = strpr;

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

sim_struct.expe = expe;     % store experimental structure in model_sim structure

% Target parameters for experiment 
omin    = -mtgt-49;         % min value of outcomes
omax    = 100-mtgt-51;      % max value of outcomes
a       = stgt/sgen;        % slope of linear transformation aX+b
b       = mtgt - a*mgen;    % intercept of linear transf. aX+b

% Kalman Filter variables
mt = nan(nb/3,nt,ns,nc);    % posterior mean
vt = nan(nb/3,nt,ns,nc);    % posterior variance
kt = zeros(nb/3,nt,ns,nc);  % kalman gain
vs = stgt^2;                % sampling variance
vd = (alpha/(1-alpha))^2;              % (perceived) process noise as multiplicative factor of vs

% Structure learning contribution
%   4-value variable (to determine contribution of structure learning) 
%     whose values are log odds from SL
p_sl    = nan(4,nc);    % probability of choice A 
s       = zeros(4,nc);  % logit of p_sl

% Value structures
p_rl = zeros(nb/3,nt,ns,nc);% p(shape A) from RL
q  = zeros(nb/3,nt,ns,nc);  % log odds from RL
mp = nan(nb/3,1,ns,nc);     % subjective probability of shape A being right
dv = nan(nb/3,nt,ns,nc);    % decision variable
oc = nan(nb/3,nt,ns,nc);    % outcomes

ib_c        = zeros(1,3);   % index of condition blocks
pos_shape   = zeros(3,1);   % tracks the previous good shape for outcome signage
out_mult    = 1;            % multiplier on the outcome for signage based on condition


iqrtr    = 0;               % index on quarter
for ib = 1:nb
    
    shapeset	= expe(ib).shape;
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
    ib_c(ic) = ib_c(ic) + 1;
    
    % track quarter index
    if mod(ib,12) == 1
        iqrtr = iqrtr + 1;
    end
    
    % outcome data structure

    %pos_shape(ic)   = shapeset(1);
    p_sl(ic,iqrtr)  = strpr;

    % Set structure learning contribution
    s(ic,iqrtr) = log(p_sl(ic,iqrtr)) - log(1-p_sl(ic,iqrtr));
    
    % Trial loop
    for it = 1:nt
        
        oc_temp = round((out_mult*expe(ib).blck(it).*a+b));
        oc_temp(oc_temp>omax) = omax;
        oc_temp(oc_temp<omin) = omin;
        oc(ib_c(ic),it,:,ic)  = oc_temp;
        expe(ib).blck_trn(it) = oc_temp;
        
        if it == 1
            % Initialization of Kalman Filter variables
            mt(ib_c(ic),it,:,ic)    = 0;     % KF posterior mean = 0 (initial)
            vt(ib_c(ic),it,:,ic)    = 1e2;   % KF posterior variance = 100 (initial)
            kt(ib_c(ic),it,:,ic)    = 0;     % KF gain = 0 (initial)
        else
            kt(ib_c(ic),it,:,ic)    = vt(ib_c(ic),it-1,:,ic)./(vt(ib_c(ic),it-1,:,ic)+vs); 	% Kalman gain update
            mt(ib_c(ic),it,:,ic)  	= mt(ib_c(ic),it-1,:,ic)+(oc(ib_c(ic),it-1,:,ic)...     % Mean estimate
                                     -mt(ib_c(ic),it-1,:,ic)).*kt(ib_c(ic),it,:,ic).*(1+randn(1,1,ns)*zeta);
            vt(ib_c(ic),it,:,ic)	= (1-kt(ib_c(ic),it,:,ic)).*vt(ib_c(ic),it-1,:,ic);	% Covariance noise update
        end
        p_rl(ib_c(ic),it,:,ic)  = 1-normcdf(0,mt(ib_c(ic),it,:,ic),sqrt(vt(ib_c(ic),it,:,ic)));  % Update contribution from RL
        q(ib_c(ic),it,:,ic)     = log(p_rl(ib_c(ic),it,:,ic)) - log(1-p_rl(ib_c(ic),it,:,ic));   % Convert to logit
        mp(ib_c(ic),it,:,ic)    = (1+exp(-(q(ib_c(ic),it,:,ic)+s(ic,iqrtr)))).^-1;               % Estimate probability of the shape 
        
        w_dv = round(mp(ib_c(ic),it,:,ic)); % convert probabilities to argmax
        %w_dv = mp(ib_c(ic),it,:,ic);         % softmaxed
        
        % Choice 
        dv(ib_c(ic),it,ic)   = datasample(shapeset,1,'Weights',[w_dv 1-w_dv]);
        
      % testing  
        if dv(ib_c(ic),it,ic) ~= shapeset(1)
            expe(ib).blck_trn(it) = round((-expe(ib).blck(it).*a+b));
        end
        
        % Extrapolate covariance for next step
        vt(ib_c(ic),it,:,ic)    = vt(ib_c(ic),it,:,ic) + vd; % covariance extrapolation (process noise included i.e. vd)
        
        
    end % trial loop
    
    resp = dv(ib_c(ic),:,:,ic);
    resp(resp == shapeset(2))=0;
    resp(resp ~= 0) = 1;
    resp(resp == 0) = 2;
    
    expe(ib).resp  = resp;
    
end % block loop 

sim_struct.expe = expe;

sim_struct.sim.q    = q;
sim_struct.sim.s    = s;
sim_struct.sim.dv   = dv;
sim_struct.sim.kt   = kt;
sim_struct.sim.mp   = mp;
sim_struct.sim.mt   = mt;
sim_struct.sim.vt   = vt;
sim_struct.sim.p_rl = p_rl;
sim_struct.sim.p_sl = p_sl;

out.sim_struct = sim_struct;

end