function objFn = pf_model2_traj_rlvsl(params)

% Sequential MC Particle Filter on 2 parameter model (model 2) for RLvSL
% (2 parameter model with structure learning fitted directly over each quarter)
%
% Input:    params 
%            Generative parameter values to be used in simulating the data
%           params : [zeta kappa strpr] for 
%
% Global:
%           data
%            Experimental structure containing simulated/real data
%           datasource
%            String of specifying source of the data
%           : 'sim' if simulated data
%           : 'hum' if human data
%           fitcond
%            Cell array of string specify which condition to fit
%           : 'rep','alt','rnd' 
%           qrtr
%            Integer specifying experimental quarter to analyse
%           : 1, 2, 3, 4
%
% Output:   objFn
%           : log(sum(exp(p(params|data))))
%
% Version:  -objective function calculation is based on accuracy of trajectory
%           -builds upon the previous versions of sim_pf_model_rlvsl.m. 
%           -considers that we are no longer using the structure learning 
%               aspect for the 'random' condition. 
%           -fits a unique set of parameters and the value of the contribution 
%               from structure learning directly over each quarter given a condition.
%           -will only fit 1 experimental condition for any given run.
%
% Notes:    The global variables are forced due to the nature of the BADS optimizer
%           being used to find the optimum.
%
%           - Feb 11 2020

% Global variables
global qrtr
global data;
global datasource;
global nparticles;
global fitcond;

% Generative parameters
kappa   = params(1); 
zeta    = params(2); 
strpr   = params(3); 

% Structure containing experimental and behavioral data

if ~ismember(fitcond{1},{'rep','alt','rnd','all'})
    error('Unexpected input. Enter either ''rep'',''alt'',''rnd'', or ''all''.');
end

switch fitcond{1}
    case 'rnd' % random across successive blocks
        ic = 3;
    case 'alt' % always alternating
        ic = 2;
    case 'rep' % always the same
        ic = 1;
end

% Particle Filter parameters and structures
np = nparticles; % number of particles to maintain during simulation

global cfg_tgt;
% Organize necessary information from experiment / response structures
if strcmpi(datasource,'sim')
    sim  = data.sim;
    expe = data.expe;
    dv = sim.dv;    % decisions from simulated data
    mtgt    = sim.cfg_tgt.mtgt;     % Target mean of difference distribution
    stgt    = sim.cfg_tgt.stgt;     % Target std of difference distribution
else
    expe = data.expe;
    cfg     = expe(1).cfg;
    ib_c    = ones(3,1);
    for ib = 4:cfg.nbout+3
        ctype = expe(ib).type;
        switch ctype
            case 'rnd' % random across successive blocks
                ic = 3;
            case 'alt' % always alternating
                ic = 2;
            case 'rep' % always the same
                ic = 1;
        end
        dv(ib_c(ic),:,ic) = expe(ib).resp;
        ib_c(ic) = ib_c(ic) + 1;
    end
    mtgt = cfg_tgt.mtgt;     % Target mean of difference distribution
    stgt = cfg_tgt.stgt;     % Target std of difference distribution
end
dv = reshape(dv,[16 16 3]); % HARD CODED; may need fixing in later versions

% Experimental constants
nt      = expe(1).cfg.ntrls;    % Number of trials per block
nb      = expe(1).cfg.nbout;    % Number of total blocks
nc      = 3;                    % Number of conditions
nb_q    = nb/nc/4;              % Number of blocks per condition per quarter
mgen    = expe(1).cfg.mgen;     % Generative distribution mean
sgen    = expe(1).cfg.sgen;     % Generative distribution std

% Objective function structures

sim_choices = nan(nb_q,nt,np);  % Results from softmax decision of particles
lfn         = nan(np,nb_q);     % Likelihood function

% Kalman Filter variables
mt = nan(nb_q,nt,np);        % Posterior mean
vt = nan(nb_q,nt,np);        % Posterior variance
kt = zeros(nb_q,nt,np);      % Kalman gain
vs = stgt^2;                 % Sampling variance
vd = kappa*vs;               % Process noise (perceived) as multiplicative factor of vs

% Structure learning contribution
p_sl = params(3);

% Particle variables leading to belief and decision
wdv = nan(nb_q,nt,np);      % Subjective probability of shape A being CHOSEN (action)
prl = zeros(nb_q,nt,np);    % p(shape A) from SL
q   = zeros(nb_q,nt,np);    % Log odds from RL
mp  = nan(nb_q,nt,np);      % Subjective probability of shape A being right


% Outcome values
oc      = nan(nb_q,nt,np);      % outcomes
omid    = 50;                   % outcome neutral value (midpoint)
omin    = -mtgt-(omid-1);       % min value of outcomes
omax    = 100-mtgt-(omid+1);    % max value of outcomes
a       = stgt/sgen;            % slope of linear transformation aX+b
b       = mtgt - a*mgen;        % intercept of linear transf. aX+b

ib_c        = zeros(1,3);       % relative index of blocks within a given condition
out_mult    = 1;                % multiplier on the outcome for signage based on condition

fprintf('Kappa: %.2f, Zeta: %.2f, Structure learned prior strength: %.2f\n',params(1),params(2),params(3));

ib_q = 1;       % relative indexing of blocks within a quarter within condition
for ib = 1:nb   % absolute indexing of blocks
    
    % Only consider the desired quarter
    if floor((ib/(nb+1)*4)+1) ~= qrtr
        continue;
    end
    
    if ~strcmpi(datasource,'sim')
        ib = ib+3;
    end
    
    % Only consider the desired condition
    ctype = expe(ib).type;
  
    if ~ismember(ctype,fitcond)
        continue;
    end
    
    % outcome data structure
    shapeset = expe(ib).shape;
    if ib_q == 1
        pos_shape   = shapeset(1);
        p_sl        = strpr;
    end
    
    
    % Note: the structure-learned prior is ALWAYS for the good shape
    if ib_q >= 2
        if shapeset(1) ~= pos_shape % triggered only for alt condition
            p_sl	 = 1-strpr;
        else
            p_sl	 = strpr;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ignore for now %%%%%%%%%%%%%%%%%%%%
    
    % multiplier on the outcome for signage based on condition
    % FIX: The multiplier on the outcome needs to be be based on both the correct
    %       shape in the condition as well as the choice of the simulated model. The
    %       data structure of the multipler will need to reflect the number of
    %       simulations.
    
    %      Will need to work out something for the 1st trial response.
    %      The outcome multiplier will then be based on these responses.
    
    
    % outcomes for the given block
    % FIX: Outcomes need to be based on the response of the particle and not fixed at
    %       whatever the data saw since they are not necessarily making the same choices at
    %       each time step.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% /ignore for now %%%%%%%%%%%%%%%%%%%%
    
    % Set structure learning contribution
    s = log(p_sl) - log(1-p_sl);
    
    for it = 1:nt
        if it == 1 
            % Initialization of Kalman Filter variables
            mt(ib_q,it,:)    = 0;     % KF posterior mean = 0 (initial)
            vt(ib_q,it,:)    = 1e2;   % KF posterior variance = 100 (initial)
            kt(ib_q,it,:)    = 0;     % KF gain = 0 (initial)
        else
            
            % A1. KF update
            kt(ib_q,it,:) = vt(ib_q,it-1,:)./(vt(ib_q,it-1,:)+vs);      % Kalman gain update
            mt(ib_q,it,:) = mt(ib_q,it-1,:)+(oc(ib_q,it-1,:)...         % Mean estimate update 
                            -mt(ib_q,it-1,:)).*kt(ib_q,it,:).*(1+randn(1,1,np,1)*zeta);
            vt(ib_q,it,:) = (1-kt(ib_q,it,:)).*vt(ib_q,it-1,:);         % Covariance noise update
            
        end
        
        % Belief update (incorporate RL with SL)
        prl(ib_q,it,:)  = 1-normcdf(0,mt(ib_q,it,:),sqrt(vt(ib_q,it,:)));
        q(ib_q,it,:)    = log(prl(ib_q,it,:)) - log(1-prl(ib_q,it,:));
        
        % Decision probability
        mp(ib_q,it,:) = (1+exp(-(q(ib_q,it,:)+s))).^-1;
        
        % Decision output
        for ip = 1:np
            sim_choices(ib_q,it,ip) = randsample([1 2], 1, true, [mp(ib_q,it,ip) 1-mp(ib_q,it,ip)]);
        end
        
        if it < nt
            % Resulting outcomes based on simulation decision
            choice2outmult  = sim_choices(ib_q,it,:);
            choice2outmult  = -(choice2outmult-1.5)*2; % map 2 to -1; 1 t 1
            
            % Calculate correct outcome multiplier for the KF update on the next step
            oc_temp                 = round((choice2outmult.*expe(ib).blck(it).*a+b));
            oc_temp(oc_temp>omax)   = omax;
            oc_temp(oc_temp<omin)   = omin;
            oc(ib_q,it,:)           = oc_temp;
        end
        
        % Covariance extrapolation update
        vt(ib_q,it,:) = vt(ib_q,it,:) + vd;

    end % trial loop
    
    % Likelihood function calculation
    lfn(:,ib_q) = sum(double(eq(reshape(sim_choices(ib_q,:,:),[np nt]),repmat(dv(ib_q,:,ic),[np 1]))),2)/nt;

    % relative indexing of blocks based on condition
    ib_q = ib_q + 1;
    if ~strcmpi(datasource,'sim')
        ib = ib-3;
    end
    
end % block loop

objFn = log(sum(sum(exp(lfn))));
fprintf('ObjFn = %d \n',objFn);


end

