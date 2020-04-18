function objFn = pf_model_rlvsl(params)

% Sequential MC Particle Filter on 4 parameter model (model 1) for RLvSL
% (4 parameter model with structure learning as updates of a Beta distribution)
%
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
%           : log(sum(exp(p(params|data))))
%
% Version:  This version of pf_model_rlvsl builds upon the previous versions of
%           sim_pf_model_rlvsl.m. This version considers that we are no longer using
%           the structure learning aspect for the 'random' condition.
%
% Notes:    The global variables are forced due to the nature of the BADS optimizer
%           being used to find the optimum.
%
%           - Jan 23 2020

global data;
global datasource;
% Conditions to fit
global fitcond;
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

% Structure containing experimental and behavioral data
%global data;
sim  = data.sim;
expe = data.expe;

% Organize necessary information from experiment / response structures
dv = [];
if strcmpi(datasource,'sim')
    dv = sim.dv;    % decisions from simulated data
else
    dv = nan;       % to be determined based on human data
end

% Experimental constants
nt      = expe(1).cfg.ntrls;    % Number of trials per block
nb      = expe(1).cfg.nbout;    % Number of total blocks
nc      = 3;                    % Number of conditions
nb_c    = nb/nc;                % Number of blocks per condition
mgen    = expe(1).cfg.mgen;     % Generative distribution mean
sgen    = expe(1).cfg.sgen;     % Generative distribution std
mtgt    = sim.cfg_tgt.mtgt;     % Target mean of difference distribution
stgt    = sim.cfg_tgt.stgt;     % Target std of difference distribution

% Objective function structures
lse     = nan(nb_c,nt,nc);      % LogSumExp of the likelihoods at each trial

% Kalman Filter variables
mt = nan(nb_c,nt,np,nc);        % Posterior mean
vt = nan(nb_c,nt,np,nc);        % Posterior variance
kt = zeros(nb_c,nt,np,nc);      % Kalman gain
vs = stgt^2;                    % Sampling variance
vd = kappa*vs;                  % Process noise (perceived) as multiplicative factor of vs

% Particle variables leading to belief and decision
mp  = nan(nb_c,nt,np,nc);       % Subjective probability of shape A being CORRECT (belief)
wdv = nan(nb_c,nt,np,nc);       % Subjective probability of shape A being CHOSEN (action)
abs = zeros(nb_c,2,nt,np,nc);   % Last-trial beta distribution parameters for the final particles
psl = zeros(nb_c,nt,np,nc);     % p(shape A) from RL
prl = zeros(nb_c,nt,np,nc);     % p(shape A) from SL
s   = zeros(nb_c,nt,np,nc);     % Log odds from SL
q   = zeros(nb_c,nt,np,nc);     % Log odds from RL

% Particle ancestors
anc = nan(nb_c,nt,np,nc); % ancestor registry (an ancestor is the index of particle at t-1 relative to current t)
wts = nan(nb_c,nt,np,nc); % weights for particle propagation
% KF
mt_anc  = nan(nb_c,nt,np,nc);   
vt_anc  = nan(nb_c,nt,np,nc);   
kt_anc  = zeros(nb_c,nt,np,nc); 
%
abs_anc = zeros(nb_c,2,nt,np,nc);
s_anc   = zeros(nb_c,nt,np,nc);
%
mt_protoanc = nan(nb_c,2,nt,np,nc); % sign of KF tracked means for beta param calculations

% Outcome values
oc      = nan(nb_c,nt,np,nc);   % outcomes
omid    = 50;                   % outcome neutral value (midpoint)
omin    = -mtgt-(omid-1);       % min value of outcomes
omax    = 100-mtgt-(omid+1);    % max value of outcomes
a       = stgt/sgen;        % slope of linear transformation aX+b
b       = mtgt - a*mgen;        % intercept of linear transf. aX+b

% Indexing and interval tracking
ib_c        = zeros(1,3);       % index of blocks within a given condition
decay_trckr = zeros(nb/3,3);    % track blocks of a certain condition for decay (not in use)
pos_shape   = zeros(3,1);       % tracks the previous good shape for outcome signage
out_mult    = 1;                % multiplier on the outcome for signage based on condition

fprintf('Rescaled Beta: %.2f, Gamma: %.2f, Kappa: %.4f, Zeta: %.2f ',params(1),params(2),params(3),params(4));

% Initialize particle ancestor and 1st weight
anc(1,1,:,:) = 0;               % each first particle is its own ancestor (but value shall be 0)
wts(1,1,:,:) = 1/np;            % weights = 1/np since all have equal weight in beginning

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
    ib_c(ic) = ib_c(ic) + 1;

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

    
    % Beginning of 1st and 2nd blocks - Initialization
    if ib_c(ic) < 3
        % Initialize particles
        psl(ib_c(ic),1,:,ic)    = 1*0.5; % p(shape A) from RL = 0.5 (initial)
        prl(ib_c(ic),1,:,ic)    = 1*0.5; % p(shape A) from SL = 0.5 (initial)
        abs(ib_c(ic),:,1,:,ic)  = 1;     % Beta distrib. parameters = 1 (initial)
    
    % The beginning of the rest of the blocks 
    else
        % No SL update for the 'rnd' condition
        if ic == 3
            psl(ib_c(ic),1,:,ic)    = 1*0.5; % maintain SL data structures for 'rnd'
            q(ib_c(ic),1,:,ic)      = 0;     %% for reducing special cases in calculation
            abs(ib_c(ic),:,1,:,ic)  = 1;     %%% this should not have a huge effect in the filtering
        else
            % Obtain switch/stay from mt_protoanc
            signbminus2 = sign(mt_protoanc(ib_c(ic)-1,1,end,:,ic)); % final tracked mean 2 blocks ago
            signbminus1 = sign(mt_protoanc(ib_c(ic)-1,2,end,:,ic)); % final tracked mean 1 block ago
            % Convert to Beta distribution parameters
            abs_a = double(bsxfun(@eq, signbminus2, signbminus1));
            abs_b  = double(~bsxfun(@eq, signbminus2, signbminus1));
            % Apply decay
            abs(ib_c(ic),1,1,:,ic) = abs_anc(ib_c(ic)-1,1,end,:,ic) + abs_a.*(1-gamma); % α(i) = α(i-1) + (1-γ)*a 
            abs(ib_c(ic),2,1,:,ic) = abs_anc(ib_c(ic)-1,1,end,:,ic) + abs_b.*(1-gamma); % β(i) = β(i-1) + (1-γ)*b
            
            % Calculate SL contribution for the block
            for ip = 1:np
                % Reconsiderations:
                % May need to think more about whether this sign is based on mt or mp 
                %   The sign of the mean tracked value?
                %   The sign based on the value of mp?
                if sign(mt_anc(ib_c(ic)-1,end,ip,ic)) == 1
                    psl(ib_c(ic),1,ip,ic) = abs(ib_c(ic),1,1,ip,ic)./(abs(ib_c(ic),1,1,ip,ic)+abs(ib_c(ic),2,1,ip,ic));
                else
                    psl(ib_c(ic),1,ip,ic) = 1-(abs(ib_c(ic),1,1,ip,ic)./(abs(ib_c(ic),1,1,ip,ic)+abs(ib_c(ic),2,1,ip,ic)));
                end
            end
        end
    
        mt_protoanc(ib_c(ic),1,1,:,ic) = mt_protoanc(ib_c(ic)-1,2,end,:,ic);
    
    end
    
    % Trial-by-trial dynamics
    for it = 1:nt
        % Decision dynamics 
        %fprintf('ib_c = %d, it = %d, ic = %d\n',ib_c(ic),it,ic); % debug
        if it == 1 % 1st trials
            
            % 1st trial incorporation of SL priors
            s(ib_c(ic),1,:,ic) = log(psl(ib_c(ic),1,:,ic)) - log(1-psl(ib_c(ic),1,:,ic));
            
            % Initialization of Kalman Filter variables
            mt(ib_c(ic),it,:,ic)     = 0;     % KF posterior mean = 0 (initial)
            vt(ib_c(ic),it,:,ic)     = 1e2;   % KF posterior variance = 100 (initial)
            kt(ib_c(ic),it,:,ic)     = 0;     % KF gain = 0 (initial)
            
        else % it > 1
            s(ib_c(ic),it,:,ic)  = s_anc(ib_c(ic),it-1,:,ic);
            % A1. KF update
            % the ancestors are from t-1 on the current condition block
%            disp('KF update step');
%            disp(['vt_anc: ' num2str(reshape(vt_anc(ib_c(ic),it-1,:,ic),1,np))])
            kt(ib_c(ic),it,:,ic) = vt_anc(ib_c(ic),it-1,:,ic)./(vt_anc(ib_c(ic),it-1,:,ic)+vs);  % Kalman gain update
%            disp(['mt_anc: ' num2str(reshape(mt_anc(ib_c(ic),it-1,:,ic),1,np))])
%            disp(['kt: ' num2str(reshape(kt(ib_c(ic),it,:,ic),1,np))])
            mt(ib_c(ic),it,:,ic) = mt_anc(ib_c(ic),it-1,:,ic)+(oc(ib_c(ic),it-1,:,ic)...         % Mean estimate update 
                                  -mt_anc(ib_c(ic),it-1,:,ic)).*kt(ib_c(ic),it,:,ic).*(1+randn(1,1,np,1)*zeta);
            vt(ib_c(ic),it,:,ic) = (1-kt(ib_c(ic),it,:,ic)).*vt_anc(ib_c(ic),it-1,:,ic);         % Covariance noise update

        end
        
        % A2. Contribution of RL toward belief
        prl(ib_c(ic),it,:,ic)	= 1-normcdf(0,mt(ib_c(ic),it,:,ic),sqrt(vt(ib_c(ic),it,:,ic))); % RL contribution update
        q(ib_c(ic),it,:,ic)     = log(prl(ib_c(ic),it,:,ic)) - log(1-prl(ib_c(ic),it,:,ic));    % Convert to logit
        
        % A3. Incorporate block-constant SL with the trial-by-trial RL update into
        % decision belief
        mp(ib_c(ic),it,:,ic) = (1+exp(-(q(ib_c(ic),it,:,ic)+s(ib_c(ic),1,:,ic)))).^-1;          % p(shape A) correct 
%        disp(['mp: ' num2str(reshape(mp(ib_c(ic),it,:,ic),1,np))]); % debug
        
        % Particle filter dynamics 
     
        % B1. Weight the particles based on correspondence with data 
        % Probability of reference choice taken by particles via SOFTMAX weighting
        wdv(ib_c(ic),it,:,ic)	= 1/(1+exp(-beta*(2*mp(ib_c(ic),it,:,ic)-1))); % (p-(1-p)) = 2p-1
        if dv(ib_c(ic),it,1,ic) == shapeset(1)      % sim response = correct shape
            
        elseif dv(ib_c(ic),it,1,ic) == shapeset(2)  % sim response = incorrect shape
            wdv(ib_c(ic),it,:,ic) = 1-wdv(ib_c(ic),it,:,ic); 
        else
            error('Unexpected sim response. Experiment set = %d %d; Simulation choice = %d',shapeset(1), shapeset(2), dv(ib_c(ic),it,1,ic));
        end
        
        %disp(['wdv: ' num2str(reshape(wdv(ib_c(ic),it,:,ic),1,np))]) % debug
        
        % B2. Update objective function (log-sum-exp)
        lse(ib_c(ic),it,ic) = log(sum(exp(wdv(ib_c(ic),it,:,ic))));
        
      % Filtering step
        %fprintf('Filtering step: ic=%d ib_c=%d it=%d\n',ic,ib_c(ic),it); % debug
        % B3. Assign weights for propagation (log-softmax)
        wts(ib_c(ic),it,:,ic)   = log(exp(wdv(ib_c(ic),it,:,ic))/sum(exp(wdv(ib_c(ic),it,:,ic))));
        wts_sum                 = sum(wts(ib_c(ic),it,:,ic));
        wts(ib_c(ic),it,:,ic)   = wts(ib_c(ic),it,:,ic)/wts_sum;
        
        % B4. Convert weights to number of descendants for each particle
        
        % FIX : There is a major issue in this part...
        rs_mult = round(reshape(wts(ib_c(ic),it,:,ic)*np,1,np)); % resampling multiplier = weight * nparticles
        
        % B5. Rank the weights from highest to lowest
        [wts_anc,index_anc] = sort(rs_mult,'Descend');
        
        % B6. Resample ancestors by decreasing order on weights
        rs_part_ct = 0; % count particles resampled
        
        % Before sampling on the weighted particles...
        % Avoid sample impoverishment (i.e. if replenishment particles < max number particles; from rounding)
        if sum(rs_mult) < np
            %disp('rs_mult < np triggered'); % debug
            n_impov = np - sum(rs_mult);    % calculate number of missing particles
            
            % Perform multinomial resampling based on the precalculated weights
            impovs   = datasample(index_anc,n_impov,'Weights',wts_anc); % values of index_anc that are to be resampled
            %disp(['n_impov = ' num2str(impovs)]); %debug
            rs_range = rs_part_ct+1:n_impov;                            % index range to be replenished
            
            mt_anc(ib_c(ic),it,rs_range,ic) = mt(ib_c(ic),it,impovs,ic);
            vt_anc(ib_c(ic),it,rs_range,ic) = vt(ib_c(ic),it,impovs,ic);
            kt_anc(ib_c(ic),it,rs_range,ic) = kt(ib_c(ic),it,impovs,ic);
            s_anc(ib_c(ic),it,rs_range,ic)  = s(ib_c(ic),it,impovs,ic);
            
            if it == 1
                abs_anc(ib_c(ic),:,it,rs_range,ic) = abs(ib_c(ic),:,it,impovs,ic);
            else
                abs_anc(ib_c(ic),:,it,rs_range,ic) = abs_anc(ib_c(ic),:,it-1,impovs,ic);
            end
            
            if ~(ib_c(ic) == 1) && ~(it == 1)
                mt_protoanc(ib_c(ic),:,it,rs_range,ic) = mt_protoanc(ib_c(ic),:,it-1,impovs,ic);
            end
            
            anc(ib_c(ic),it,rs_part_ct+1:n_impov,ic) = impovs; % track ancestor indices
            
            % assign their ancestors to the proper indices in the data structures
            rs_part_ct = rs_part_ct + n_impov; % update assignment position
        end
        
        % Residual sampling on the weighted particles
        for ia = 1:length(index_anc)
            
            % Ignore the particles that have no weight
            if rs_mult(index_anc(ia)) == 0
                continue;
            else
                % Avoid oversampling (i.e. if replenishment particles > max number particles
                %                          then truncate number to be replenished)
                if rs_part_ct + rs_mult(index_anc(ia)) > np 
                    %disp('oversampling condition triggered'); % debug
                    rs_mult(index_anc(ia)) = np-rs_part_ct;
                end

                rs_range = rs_part_ct+1:rs_part_ct+rs_mult(index_anc(ia)); % index range to be replenished

                % Assign weighted particles as ancestors
                s_anc(ib_c(ic),it,rs_range,ic)  = s(ib_c(ic),it,index_anc(ia),ic);
                mt_anc(ib_c(ic),it,rs_range,ic) = mt(ib_c(ic),it,index_anc(ia),ic);
                vt_anc(ib_c(ic),it,rs_range,ic) = vt(ib_c(ic),it,index_anc(ia),ic);
                kt_anc(ib_c(ic),it,rs_range,ic) = kt(ib_c(ic),it,index_anc(ia),ic);

                % Propagate the beta parameters of the block
                if it == 1
                    abs_anc(ib_c(ic),1,it,rs_range,ic) = abs(ib_c(ic),1,it,index_anc(ia),ic);
                    abs_anc(ib_c(ic),2,it,rs_range,ic) = abs(ib_c(ic),2,it,index_anc(ia),ic);
                else
                    abs_anc(ib_c(ic),1,it,rs_range,ic) = abs_anc(ib_c(ic),1,it-1,index_anc(ia),ic);
                    abs_anc(ib_c(ic),2,it,rs_range,ic) = abs_anc(ib_c(ic),2,it-1,index_anc(ia),ic);
                end

                % Propagate last KF tracking signs of the protoancestor
                if ~(ib_c(ic) == 1) && ~(it == 1)
                    mt_protoanc(ib_c(ic),1,it,rs_range,ic) = mt_protoanc(ib_c(ic),1,it-1,index_anc(ia),ic);
                    mt_protoanc(ib_c(ic),2,it,rs_range,ic) = mt_protoanc(ib_c(ic),2,it-1,index_anc(ia),ic);
                end

                % Track indices of the ancestor (if it > 1, current it; else ib_c-1, it=end)
                anc(ib_c(ic),it,rs_range,ic) = index_anc(ia); % track ancestor indices

                % Update resampled particles count / assignment position
                rs_part_ct = rs_part_ct + rs_mult(index_anc(ia));
            end
        end
        
        % Last trial operations
        if it == nt
            % If the 1st block instance of a given condition
            if ib_c(ic) == 1
                % Initialize mt_protoanc(ib_c,1,it,:,ic) = mt_anc from above;
                mt_protoanc(ib_c(ic),1,it,:,ic) = mt_anc(ib_c(ic),it,:,ic);
            % If the 2nd block instance of a given condition 
            elseif ib_c(ic) == 2
                % Initialize mt_protoanc(ib_c,2,it,:,ic)
                mt_protoanc(ib_c(ic),2,it,:,ic) = mt_anc(ib_c(ic),it,:,ic);
            end
        end
        
        % KF Covariance extrapolation update
        vt_anc(ib_c(ic),it,:,ic)    = vt_anc(ib_c(ic),it,:,ic) + vd ;
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

if strcmpi(fitcond{1},'all')
    objFn = -sum(sum(sum(lse(:,:,:))));
else
    objFn = -sum(sum(lse(:,:,ic)));
end

if objFn == Inf || objFn == -Inf
    fprintf(objFn);
    disp('objFn overflow issues');
end
fprintf('LSE = %d \n',objFn);

end