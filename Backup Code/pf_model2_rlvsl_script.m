% Script
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
% Version:  -builds upon the previous versions of sim_pf_model_rlvsl.m. 
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

clear all;
% Generative parameters
% trial-to-trial RL difficulty calculation
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

cfg_tgt = struct;
cfg_tgt.mtgt    = 5;        % difference between the higher mean and 50
cfg_tgt.stgt    = sig_opti;

cfg_mod.nsims   = 1;
cfg_mod.kappa   = .01;
cfg_mod.zeta    = .6;
cfg_mod.strpr   = [.01 .6 .7 .8; ... % structure learned priors on the CORRECT shape
                   .5 .6 .7 .8;     % note: this should be a single value on the real fitter
                   .5 .5 .5 .5]; 

% FIX: the structure needs to be made outside the function to remain static while
%       searching parameters for a single simulation generation
sim_struct = gen_sim_model2_rlvsl(cfg_tgt,cfg_mod);

%%%%%%%%%% remove above stuff when porting to function
params(1) = .01; %cfg_mod.kappa;
params(2) = .6; %cfg_mod.zeta;
params(3) = .9; %cfg_mod.strpr(1);
params = params(1:3);

kappa   = params(1);
zeta    = params(2);
strpr   = params(3);

% Global variables for BADS
%global qrtr
qrtr = 1;
%global data;
data = sim_struct.sim_struct;

subj = 3;
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',subj,subj)
if ~exist(filename,'file')
    error('Missing experiment file!');
end
load(filename,'expe');

%global datasource;
datasource  = 'not';    %remove in function
fitcond     = {'rnd'};  % Specify the experimental condition {'rep','alt','rnd'} %remove in function

if ~ismember(fitcond,{'rep','alt','rnd','all'})
    error('Unexpected input. Enter either ''rep'',''alt'',''rnd'', or ''all''.');
end

% Particle Filter parameters and structures
%global nparticles;
nparticles = 1000;
np = nparticles; % number of particles to maintain during simulation

% Structure containing experimental and behavioral data
sim  = data.sim;

% Organize necessary information from experiment / response structures
dv = [];
if strcmpi(datasource,'sim')
    dv = sim.dv;    % decisions from simulated datai
else
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
end
dv = reshape(dv,[16 16 3]); % HARD CODED; may need fixing in later versions

% Experimental constants
nt      = expe(1).cfg.ntrls;    % Number of trials per block
nb      = expe(1).cfg.nbout;    % Number of total blocks
nc      = 3;                    % Number of conditions
nb_q    = nb/nc/4;              % Number of blocks per condition per quarter
mgen    = expe(1).cfg.mgen;     % Generative distribution mean
sgen    = expe(1).cfg.sgen;     % Generative distribution std
mtgt    = sim.cfg_tgt.mtgt;     % Target mean of difference distribution
stgt    = sim.cfg_tgt.stgt;     % Target std of difference distribution

% Objective function structures
sim_choices = nan(nb_q,nt,np);  % Results from softmax decision of particles
lfn         = nan(nb_q,nt);     % Likelihood function

% Kalman Filter variables
mt = nan(nb_q,nt,np);        % Posterior mean
vt = nan(nb_q,nt,np);        % Posterior variance
kt = zeros(nb_q,nt,np);      % Kalman gain
vs = stgt^2;                 % Sampling variance
vd = kappa*vs;               % Process noise (perceived) as multiplicative factor of vs

% Structure learning contribution
p_sl    = params(3);

% Particle variables leading to belief and decision
wdv = nan(nb_q,nt,np);      % Subjective probability of shape A being CHOSEN (action)
prl = zeros(nb_q,nt,np);    % p(shape A) from SL
q   = zeros(nb_q,nt,np);    % Log odds from RL
mp  = nan(nb_q,nt,np);      % Subjective probability of shape A being right

% Filtering variables
rs_mult = zeros(nb_q,nt,np);    % resampling multipliers
mt_anc  = nan(nb_q,nt,np);
vt_anc  = nan(nb_q,nt,np);
kt_anc  = nan(nb_q,nt,np);
anc     = nan(nb_q,nt,np);

% Outcome values
oc      = nan(nb_q,nt,np);      % outcomes
omid    = 50;                   % outcome neutral value (midpoint)
omin    = -mtgt-(omid-1);       % min value of outcomes
omax    = 100-mtgt-(omid+1);    % max value of outcomes
a       = sig_opti/sgen;        % slope of linear transformation aX+b
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
    switch ctype
        case 'rnd' % random across successive blocks
            ic = 3;
        case 'alt' % always alternating
            ic = 2;
        case 'rep' % always the same
            ic = 1;
    end
    
    if ~ismember(ctype,fitcond)
        continue;
    end
    
    % outcome data structure
    shapeset = expe(ib).shape;
    if ib_q == 1
        pos_shape   = shapeset(1);
        p_sl        = strpr;
    end
    
    % multiplier on the outcome for signage based on condition 
    if ib_q >= 2
        if shapeset(1) ~= pos_shape % triggered only for alt condition
            p_sl	 = 1-strpr; 
            out_mult = -1;
        else
            p_sl	 = strpr;
            out_mult = 1;
        end
    end
    
    % outcomes for the given block
    oc_temp               = round((out_mult.*expe(ib).blck.*a+b)).*ones(1,nt,np);
    oc_temp(oc_temp>omax) = omax;
    oc_temp(oc_temp<omin) = omin;
    oc(ib_q,:,:)            = oc_temp;
    
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
            kt(ib_q,it,:) = vt_anc(ib_q,it-1,:)./(vt_anc(ib_q,it-1,:)+vs);  % Kalman gain update
            mt(ib_q,it,:) = mt_anc(ib_q,it-1,:)+(oc(ib_q,it-1,:)...         % Mean estimate update 
                            -mt_anc(ib_q,it-1,:)).*kt(ib_q,it,:).*(1+randn(1,1,np,1)*zeta);
            vt(ib_q,it,:) = (1-kt(ib_q,it,:)).*vt_anc(ib_q,it-1,:);         % Covariance noise update
            
        end
        
        % Belief update (incorporate RL with SL)
        prl(ib_q,it,:)  = 1-normcdf(0,mt(ib_q,it,:),sqrt(vt(ib_q,it,:)));
        q(ib_q,it,:)    = log(prl(ib_q,it,:)) - log(1-prl(ib_q,it,:));
        
        % Decision probability
        mp(ib_q,it,:) = (1+exp(-(q(ib_q,it,:)+s))).^-1;
        
        % Weight particles based on correspondence with data (on softmax)
        if dv(ib_q,it,ic) == shapeset(1) % Correct response in data
            wdv(ib_q,it,:)  = exp(mp(ib_q,it,:));
            dmp             = mp(ib_q,it,:);
        else
            wdv(ib_q,it,:)  = exp(1-mp(ib_q,it,:));
            dmp             = 1-mp(ib_q,it,:);
        end
        
        % Update objective function
        for ip = 1:np
            sim_choices(ib_q,it,ip) = randsample([1 0], 1, true, [dmp(ip) 1-dmp(ip)]);
        end
        lfn(ib_q,it) = sum(sim_choices(ib_q,it,ip))/np;
        
        wdv(ib_q,it,:) = log(wdv(ib_q,it,:))/log(sum(wdv(ib_q,it,:))); % weights on logsumexp
        wdv(ib_q,it,:) = wdv(ib_q,it,:)/sum(wdv(ib_q,it,:));           % convert to normalized weights 
        
        % Filter particles
        
            % Convert weights to number of descendants for each particle
            rs_mult(ib_q,it,:) = round(wdv(ib_q,it,:)*np);
            
            % Rank the weights from highest to lowest
            [wts_anc,index_anc] = sort(rs_mult(ib_q,it,:),'Descend');
            wts_anc     = reshape(wts_anc, [np 1]);
            index_anc   = reshape(index_anc, [np 1]);
            
            % Resample ancestors by decreasing order on weights
            rs_part_ct = 0; % count particles already resampled
            
            % Before sampling on the weighted particles...
            % Avoid sample impoverishment (i.e. if replenishment particles < max number particles; from rounding)
            if sum(rs_mult(ib_q,it,:)) < np
                n_impov = np - sum(rs_mult(ib_q,it,:));    % calculate number of missing particles
                
                % Perform multinomial resampling based on the precalculated weights
                impovs   = datasample(index_anc,n_impov,'Weights',wts_anc); % values of index_anc that are to be resampled
                rs_range = rs_part_ct+1:n_impov;                            % index range to be replenished
                
                mt_anc(ib_q,it,rs_range) = mt(ib_q,it,impovs);
                vt_anc(ib_q,it,rs_range) = vt(ib_q,it,impovs);
                kt_anc(ib_q,it,rs_range) = kt(ib_q,it,impovs);

                anc(ib_q,it,rs_range) = impovs;
                
                rs_part_ct = rs_part_ct + n_impov; % update assignment position
            end
            
            % Residual sampling on the weighted particles
            for ia = 1:length(index_anc)
                % Ignore the particles that have no weight
                if rs_mult(ib_q,it,index_anc(ia)) == 0
                    
                else
                    % Avoid oversampling (i.e. if replenishment particles > max number particles
                    %                          then truncate number to be replenished)
                    if rs_part_ct + rs_mult(ib_q,it,index_anc(ia)) > np 
                        rs_mult(ib_q,it,index_anc(ia)) = np-rs_part_ct;
                    end
                    rs_range = rs_part_ct+1:rs_part_ct+rs_mult(ib_q,it,index_anc(ia)); % index range to be replenished
                    
                    % Assign weighted particles as ancestors
                    mt_anc(ib_q,it,rs_range) = mt(ib_q,it,index_anc(ia));
                    vt_anc(ib_q,it,rs_range) = vt(ib_q,it,index_anc(ia));
                    kt_anc(ib_q,it,rs_range) = kt(ib_q,it,index_anc(ia));
                    
                    % Track indices of the ancestor 
                    anc(ib_q,it,rs_range) = index_anc(ia);
                    
                    % Update resampled particles count / assignment position
                    rs_part_ct = rs_part_ct + rs_mult(ib_q,it,index_anc(ia));
                end
            end
        
        % Covariance extrapolation update
        vt_anc(ib_q,it,:) = vt_anc(ib_q,it,:) + vd;

        
    end % trial loop
    
    
    % relative indexing of blocks based on condition
    ib_q = ib_q + 1;
    if ~strcmpi(datasource,'sim')
        ib = ib-3;
    end
end % block loop

objFn = -log(sum(sum(exp(lfn))));
fprintf('LSE = %d \n',objFn);



