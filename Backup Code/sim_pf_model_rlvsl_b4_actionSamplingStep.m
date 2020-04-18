function objFn = sim_pf_model_rlvsl(params)
% Sequential MC Particle Filter on 4 parameter model for RLvSL

% Input:    params 
%            Generative parameter values to be used in simulating the data
%
% Global:
%           data
%            Experimental structure containing simulated/real data
%           datasource
%            String of specifying source of the data
%           : 'sim' if simulated data
%           : 'hum' if human data
%           fitcond
%            Cell array of strings specify which conditions to fit
%           : 'rep','alt','rnd' (or any combination of the three)
%           : 'all' to fit all 
%
% Output:   objFn
%           : log(p(params|data))
%
% Version:  This version of sim_of_model_rlvsl does not sample the decision, but
%           propagation and objective function calculations are based directly from
%           the softmax probabilities. 
%           - Jan 17 2020

global fitcond;
% Conditions to fit
if ~ismember(fitcond,{'rep','alt','rnd','all'})
    error('Unexpected input. Enter either ''rep'',''alt'',''rnd'', or ''all''.');
end
if strcmpi(fitcond,'all')
    fitcond = {'rep','alt','rnd'};
end

% Particle Filter parameters and structures
global nparticles;
np = nparticles; % number of particles to maintain during simulation

% Model parameters
beta    = params(1)*100;	% softmax inv. temp
gamma   = params(2);        % decay
kappa   = params(3);        % learning rate asymptote
zeta    = params(4);        % learning noise

global data;
sim  = data.sim;
expe = data.expe;

global datasource;
% Organize necessary information from experiment / response structures
if strcmpi(datasource,'sim')
    dv = sim.dv; % decisions from simulated data
else
    % some other stuff depending on structure of human responses
    dv = nan; 
end

% Experiment variables
nt      = expe(1).cfg.ntrls;    % number of trials per block
nb      = expe(1).cfg.nbout;    % number of total blocks
nc      = 3;                    % number of conditions
nb_c    = nb/nc;                % number of blocks per condition
mgen    = expe(1).cfg.mgen;     % generative distribution mean
sgen    = expe(1).cfg.sgen;     % generative distribution std
mtgt    = sim.cfg_tgt.mtgt;     % target mean of difference distribution
stgt    = sim.cfg_tgt.stgt;     % target std of difference distribution

anc = nan(nb_c,nt,np,nc); % ancestor registry (an ancestor is the index of particle at t-1 relative to current t)
wts = nan(nb_c,nt,np,nc); % weights

%lls = nan(nb_c,nt,nc);      % log-likelihood storage
lse = nan(nb_c,nt,nc);      % LogSumExp of the likelihoods at each step
badParams = false;          % trigger to determine whether a set of fitting parameters do not correspond at all to data

% Kalman Filter variables
mt = nan(nb_c,nt,np,nc);    % posterior mean
vt = nan(nb_c,nt,np,nc);    % posterior variance
kt = zeros(nb_c,nt,np,nc);  % Kalman gain
vs = stgt^2;                % sampling variance
vd = kappa*vs;              % (perceived) process noise as multiplicative factor of vs

% Particle variables leading to belief and decision
mp  = nan(nb_c,nt,np,nc);   % subjective probability of shape A being CORRECT (belief)
wdv = nan(nb_c,nt,np,nc);   % subjective probability of shape A being CHOSEN (action)
abs = zeros(nb_c,2,nt,np,nc);  % last trial beta distribution parameters for the final particles
psl = zeros(nb_c,nt,np,nc);  % p(shape A) from RL
prl = zeros(nb_c,nt,np,nc); % p(shape A) from SL
s   = zeros(nb_c,nt,np,nc); % log odds from SL
q   = zeros(nb_c,nt,np,nc); % log odds from RL

% Particle ancestors
mp_anc	= nan(nb_c,nt,np,nc);   
wdv_anc = nan(nb_c,nt,np,nc);   
abs_anc = zeros(nb_c,2,nt,np,nc);  
psl_anc = zeros(nb_c,nt,np,nc); 
prl_anc = zeros(nb_c,nt,np,nc); 
s_anc   = zeros(nb_c,nt,np,nc); 
q_anc   = zeros(nb_c,nt,np,nc); 
mt_anc  = nan(nb_c,nt,np,nc);   
vt_anc  = nan(nb_c,nt,np,nc);   
kt_anc  = zeros(nb_c,nt,np,nc); 
mt_anc_beta = nan(nb_c,nt,np,nc);

% Outcome values
oc      = nan(nb_c,nt,np,nc);   % outcomes
omid    = 50;                   % outcome neutral value (midpoint)
omin    = -mtgt-(omid-1);       % min value of outcomes
omax    = 100-mtgt-(omid+1);    % max value of outcomes
a       = stgt/sgen;            % slope of linear transformation aX+b
b       = mtgt - a*mgen;        % intercept of linear transf. aX+b

ib_c        = zeros(1,3);       % index of condition blocks
decay_trckr = zeros(nb/3,3);    % track blocks of a certain condition for decay
pos_shape   = zeros(3,1);       % tracks the previous good shape for outcome signage
out_mult    = 1;                % multiplier on the outcome for signage based on condition

fprintf('Beta: %.2f, Gamma: %.2f, Kappa: %.4f, Zeta: %.2f',params(1),params(2),params(3),params(4));

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
    
    if ~ismember(ctype,fitcond)
        continue; % only fit the conditions desired
    end
    
    % relative indexing of blocks based on condition
    ib_c(ic)                 = ib_c(ic) + 1;
    decay_trckr(ib_c(ic),ic) = ib;
    
    % outcome data structure
    if ib < 4
        pos_shape(ic) = shapeset(1); % anchor tracking value to the 1st positive shape
    else
        % The sign of belief is absolute given a set of shapes
        if shapeset(1) ~= pos_shape(ic)
            out_mult = -1;
        else
            out_mult = 1;
        end
    end
    
    % Outcomes for the block
    oc_temp = round((out_mult.*expe(ib).blck.*a+b)).*ones(1,nt,np,1);
    oc_temp(oc_temp>omax) = omax;
    oc_temp(oc_temp<omin) = omin;
    oc(ib_c(ic),:,:,ic) = oc_temp;
    
    % Propagate ancestral beta parameters
    if ib_c(ic) > 2
        mt_anc_beta(ib_c(ic),1,:,ic) = mt_anc(ib_c(ic)-1,end,:,ic); % propagate ancestral beta parameters
    end
        
    % Prior belief and SL (Beta distribution) updating
    if ib_c(ic) < 3    % 1st and 2nd blocks (in a given condition) have flat priors
        % Initializtion of particles and their values
        anc(ib_c(ic),1,:,ic) = 0;     % each particle is its own ancestor (but value shall be 0)
        wts(ib_c(ic),1,:,ic) = 1/np;  % weights = 1/np since all have equal weight in beginning
        
        psl(ib_c(ic),1,:,ic)    = 1*0.5; % p(shape A) from RL = 0.5 (initial)
        prl(ib_c(ic),1,:,ic)    = 1*0.5; % p(shape A) from SL = 0.5 (initial)
        q(ib_c(ic),1,:,ic)      = 0;     % log odds from RL = 0 (initial)
        s(ib_c(ic),1,:,ic)      = 0;     % log odds from SL = 0 (initial)
        abs(ib_c(ic),:,1,:,ic)  = 1;     % Beta distrib. parameters = 1 (initial)
        mp(ib_c(ic),1,:,ic) 	= .5;    % p(shape A) correct = 0.5 (initial)
        
        % Initialization of Kalman Filter variables
        mt(ib_c(ic),1,:,ic) = 0;    % KF posterior mean = 0 (initial)
        vt(ib_c(ic),1,:,ic) = 1e3;  % KF posterior variance = 1000 (initial)
        kt(ib_c(ic),1,:,ic) = 0;    % KF gain = 0 (initial)
        
    elseif ib_c(ic) >= 3 % consider switch/stay-based prior from Beta distribution
        % Structure learning propagation from ancestors
        abs(ib_c(ic),:,1,:,ic)  = abs_anc(ib_c(ic)-1,:,end,:,ic);
        psl(ib_c(ic),1,:,ic)    = psl_anc(ib_c(ic)-1,end,:,ic);
        s(ib_c(ic),1,:,ic)      = s_anc(ib_c(ic)-1,end,:,ic);
        
        
        % Switch/stay calculations
        abs_temp = nan(1,2,1,np);
        
        abs_temp(1,1,1,:)       = double(bsxfun(@eq,sign(mt_anc_beta(ib_c(ic),1,:,ic)),sign(mt_anc_beta(ib_c(ic)-1,end,:,ic))));  % stays
        abs_temp(1,2,1,:)       = double(~bsxfun(@eq,sign(mt_anc_beta(ib_c(ic),1,:,ic)),sign(mt_anc_beta(ib_c(ic)-1,end,:,ic)))); % switches
        
        abs(ib_c(ic),1,1,:,ic)  = abs_anc(ib_c(ic)-1,1,end,:,ic) + abs_temp(1,1,1,:).*(1-gamma*(decay_trckr(ib_c(ic),ic)-decay_trckr(ib_c(ic)-1,ic))); % alpha++ -decay for stay
        abs(ib_c(ic),2,1,:,ic)  = abs_anc(ib_c(ic)-1,2,end,:,ic) + abs_temp(1,2,1,:).*(1-gamma*(decay_trckr(ib_c(ic),ic)-decay_trckr(ib_c(ic)-1,ic))); % beta++  -decay for switch
        
        % update p_sl where the stay/switch represents shape A
        % ** may be able to vectorize this part **
        for ip = 1:np
            if sign(mt(ib_c(ic)-1,end,ip,ic)) == 1
                psl(ib_c(ic),1,ip,ic) = abs(ib_c(ic),1,1,ip,ic)./(abs(ib_c(ic),1,1,ip,ic)+abs(ib_c(ic),2,1,ip,ic));
            else
                psl(ib_c(ic),1,ip,ic) = 1-((abs(ib_c(ic),1,1,ip,ic))./(abs(ib_c(ic),1,1,ip,ic)+abs(ib_c(ic),2,1,ip,ic)));
            end
        end
        psl_anc(ib_c(ic),1,ip,ic) = psl(ib_c(ic),1,ip,ic);
        
        % KF update
        kt(ib_c(ic),1,:,ic)     = vt_anc(ib_c(ic)-1,end,:,ic)./(vt_anc(ib_c(ic)-1,end,:,ic)+vs);                      % Kalman gain update
        mt(ib_c(ic),1,:,ic)     = mt_anc(ib_c(ic)-1,end,:,ic)+(oc(ib_c(ic)-1,end,:,ic)...                             % Mean estimate update 
                                  -mt_anc(ib_c(ic)-1,end,:,ic)).*kt_anc(ib_c(ic)-1,end,:,ic).*(1+randn(1,1,np,1)*zeta);

        vt(ib_c(ic),1,:,ic)     = (1-kt_anc(ib_c(ic)-1,end,:,ic)).*vt_anc(ib_c(ic)-1,end,:,ic);                       % Covariance noise update

        % Contribution of RL toward belief
        prl(ib_c(ic),1,:,ic)	= 1-normcdf(0,mt(ib_c(ic),1,:,ic),sqrt(vt(ib_c(ic),1,:,ic)));                         % Update contribution from RL
        q(ib_c(ic),1,:,ic)  	= log(prl(ib_c(ic),1,:,ic)) - log(1-prl(ib_c(ic),1,:,ic));                            % Convert to logit

        % Incorporate both RL and SL toward belief
        mp(ib_c(ic),1,:,ic) 	= (1+exp(-(q(ib_c(ic),it,:,ic)+s(ib_c(ic),1,:,ic)))).^-1;                             % Estimate probability of the shape
    end
    
    % subsequent trials
    for it = 1:nt
        if it > 1
            % Structure learning propagation from ancestors
            abs(ib_c(ic),:,it,:,ic) = abs_anc(ib_c(ic),:,it-1,:,ic);
            psl(ib_c(ic),it,:,ic)   = psl_anc(ib_c(ic),it-1,:,ic);
            s(ib_c(ic),it,:,ic)     = s_anc(ib_c(ic),it-1,:,ic);
            
            % KF update
            kt(ib_c(ic),it,:,ic)    = vt_anc(ib_c(ic),it-1,:,ic)./(vt_anc(ib_c(ic),it-1,:,ic)+vs);                      % Kalman gain update
            mt(ib_c(ic),it,:,ic)    = mt_anc(ib_c(ic),it-1,:,ic)+(oc(ib_c(ic),it-1,:,ic)...                             % Mean estimate update 
                                      -mt_anc(ib_c(ic),it-1,:,ic)).*kt_anc(ib_c(ic),it-1,:,ic).*(1+randn(1,1,np,1)*zeta);
            
            vt(ib_c(ic),it,:,ic)    = (1-kt_anc(ib_c(ic),it-1,:,ic)).*vt_anc(ib_c(ic),it-1,:,ic);                       % Covariance noise update

            % Contribution of RL toward belief
            prl(ib_c(ic),it,:,ic)	= 1-normcdf(0,mt(ib_c(ic),it,:,ic),sqrt(vt(ib_c(ic),it,:,ic)));                     % Update contribution from RL
            q(ib_c(ic),it,:,ic)  	= log(prl(ib_c(ic),it,:,ic)) - log(1-prl(ib_c(ic),it,:,ic));                        % Convert to logit

            % Incorporate both RL and SL toward belief
            mp(ib_c(ic),it,:,ic) 	= (1+exp(-(q(ib_c(ic),it,:,ic)+s(ib_c(ic),1,:,ic)))).^-1;                           % Estimate probability of the shape
        end
        
        % Update to maintain belief information for structure learning step 
        if it == nt % last trial in the block
            mt_anc_beta(ib_c(ic),it,:,ic) = mt(ib_c(ic),it,:,ic);
        end
        
        % Probability of action via log softmax
        if dv(ib_c(ic),it,1,ic) == shapeset(1)      % response = correct shape
            wdv(ib_c(ic),it,:,ic) = -log(1+exp(-beta*(mp(ib_c(ic),it,:,ic)-(1-mp(ib_c(ic),it,:,ic)))));
        elseif dv(ib_c(ic),it,1,ic) == shapeset(2)  % response = incorrect shape
            wdv(ib_c(ic),it,:,ic) = -log(1+exp(-beta*(-mp(ib_c(ic),it,:,ic)+(1-mp(ib_c(ic),it,:,ic)))));
        end
        wts(ib_c(ic),it,:,ic)   = wdv(ib_c(ic),it,:,ic)/sum(wdv(ib_c(ic),it,:,ic));
        lse(ib_c(ic),it,ic)     = log(sum(exp(wdv(ib_c(ic),it,:,ic))));
        
        % - Calculation of log-likelihood on trial t from softmax probabilities
        % - Conversion of softmax probabilities to weights for ancestors
        if dv(ib_c(ic),it,1,ic) == shapeset(1)      % response = correct shape
            lse(ib_c(ic),it,ic)     = log(sum(exp(wdv(ib_c(ic),it,:,ic))));
            wts(ib_c(ic),it,:,ic)   = wdv(ib_c(ic),it,:,ic)/sum(wdv(ib_c(ic),it,:,ic));
        elseif dv(ib_c(ic),it,1,ic) == shapeset(2)  % response = incorrect shape
            lse(ib_c(ic),it,ic)     = log(sum(exp(1-wdv(ib_c(ic),it,:,ic))));
            wts(ib_c(ic),it,:,ic)   = (1-wdv(ib_c(ic),it,:,ic))/sum(1-wdv(ib_c(ic),it,:,ic));
        end
        
        % Debug code
        if ismember(1,isnan(lse(ib_c(ic),it,ic)))
            error('found NaN in objective function calculation');
            
        elseif ismember(lse(ib_c(ic),it,ic), [Inf -Inf])
            objFn = 1e8;
            badParams = true;
            disp('trigger: Infinite LLH: no particle probability corresponds to action');
            break;
            
        elseif ismember(1,isnan(wts(ib_c(ic),it,:,ic))) % this means that no particle had any probability of choosing the action
            objFn = 1e8;
            badParams = true;
            disp('trigger: 0/0 Weighting: no particle probability corresponds to action');
            break;
        end
        
        %debug
        if badParams
            disp('debug trigger 1');
        end
        
        rs_mult = round(reshape(wts(ib_c(ic),it,:,ic)*np,1,np)); % resampling multiplier = weight * number of particles
        
        % Filtering step
        % 1. Rank the weights
        [wts_anc,ind_anc] = sort(rs_mult,'Descend'); % index of ancestors from highest to lowest weight
        
        % 2. Resample ancestors by decreasing order on weights
        rs_part_ct = 0; % count particles resampled
        
        % 2a. Avoid sample impoverishment (i.e. replenishment particles < number of particles)
        if sum(rs_mult) < np
            n_impov =(np-sum(rs_mult)); % amount of missing particles to be replenished
            
            % multinomial resampling based on weights
            imps = datasample(ind_anc,n_impov,'Weights',wts_anc); % the would-have-been impoverished samples 
            
            mp_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)      = mp(ib_c(ic),it,imps,ic);
            wdv_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)     = wdv(ib_c(ic),it,imps,ic);
            abs_anc(ib_c(ic),:,it,rs_part_ct+1:n_impov,ic)   = abs(ib_c(ic),:,it,imps,ic);
            psl_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)     = psl(ib_c(ic),it,imps,ic);
            prl_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)     = prl(ib_c(ic),it,imps,ic);
            s_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)       = s(ib_c(ic),it,imps,ic);
            q_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)       = q(ib_c(ic),it,imps,ic);
            mt_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)      = mt(ib_c(ic),it,imps,ic);   
            vt_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)      = vt(ib_c(ic),it,imps,ic);   
            kt_anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic)      = kt(ib_c(ic),it,imps,ic); 
            
            % copy indices of ancestors for tracking/visualisation purposes
            anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic) = imps;
            
            % Structure learning propagation from ancestor
            mt_anc_beta(ib_c(ic),it,rs_part_ct+1:n_impov,ic) = mt_anc_beta(ib_c(ic),it-1,imps,ic);
            
            rs_part_ct = rs_part_ct + n_impov; % "resampling particle count" - updates index position for step 2(b)
        end
       
        % 2b. Residual resampling
        for ia = 1:length(ind_anc)
            % Avoid oversampling (i.e. replenishment particles > number of particles)
            if rs_part_ct + rs_mult(ind_anc(ia)) > np 
                rs_mult(ind_anc(ia)) = np-rs_part_ct;
            end
            
            % copy values of ancestors for the next trial
            
            try 
                mp_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)      = mp(ib_c(ic),it,ind_anc(ia),ic);
            catch
                % Debug code
                fprintf('ic = %d, ib_c = %d, it = %d, rs_part_ct = %d, rs_mult = %d \n',ic,ib_c(ic),it,rs_part_ct,rs_mult(ind_anc(ia)));
            end
            
            wdv_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)     = wdv(ib_c(ic),it,ind_anc(ia),ic);
            abs_anc(ib_c(ic),1,it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)   = abs(ib_c(ic),1,it,ind_anc(ia),ic);
            abs_anc(ib_c(ic),2,it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)   = abs(ib_c(ic),2,it,ind_anc(ia),ic);
            psl_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)     = psl(ib_c(ic),it,ind_anc(ia),ic);
            prl_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)     = prl(ib_c(ic),it,ind_anc(ia),ic);
            s_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)       = s(ib_c(ic),it,ind_anc(ia),ic);
            q_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)       = q(ib_c(ic),it,ind_anc(ia),ic);
            mt_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)      = mt(ib_c(ic),it,ind_anc(ia),ic);   
            vt_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)      = vt(ib_c(ic),it,ind_anc(ia),ic);   
            kt_anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic)      = kt(ib_c(ic),it,ind_anc(ia),ic); 
            
            % for tracking structure learned beta distrib. parameters of ancestors on the LAST trial in a block
            if ib_c(ic) < 1 && it < nt
                disp('trigger 1'); % debug
                continue;   % these updates will crash if triggered before the mt_anc_beta structure is even filled in
            elseif it > 1
                mt_anc_beta(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic) = mt_anc_beta(ib_c(ic),it-1,ind_anc(ia),ic);
            end
%debug printout fprintf('ic = %d, ib_c = %d, it = %d, rs_part_ct = %d, rs_mult = %d \n',ic,ib_c(ic),it,rs_part_ct,rs_mult(ind_anc(ia)));
            
            % copy indices of ancestors for tracking
            anc(ib_c(ic),it,rs_part_ct+1:rs_part_ct+rs_mult(ind_anc(ia)),ic) = ind_anc(ia);
            
            % update the already resampled particles count
            rs_part_ct = rs_part_ct + rs_mult(ind_anc(ia));
            
            if rs_part_ct >= np
                break;
            end
        end
        
        %debug resampling step
        if ismember(1,isnan(mp_anc(ib_c(ic),it,:,ic)))
            error('found NaN');
        end
            
        vt_anc(ib_c(ic),it,:,ic)    = vt_anc(ib_c(ic),it,:,ic) + vd; % Covariance extrapolation with process noise
        
    end
    if badParams
        break;
    end
end

% Calculate NEGATIVE log-likelihood of parameters | data
switch fitcond{1}
    case 'rnd' % random across successive blocks
        ic = 3;
    case 'alt' % always alternating
        ic = 2;
    case 'rep' % always the same
        ic = 1;
end

if badParams
    disp('error trigger');
elseif strcmpi(fitcond{1},'all')
    objFn = -sum(sum(sum(lse(:,:,:))));
else
    objFn = -sum(sum(lse(:,:,ic)));
end

% Debug code
if objFn == Inf || objFn == -Inf
    lse
    disp('objFn overflow issues');
end
fprintf(' LSE = %d \n',objFn);

end % function