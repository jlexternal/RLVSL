% Integration of Reinforcement Learning and Structure Learning (alpha 1)
%
% This code is a first attempt at bringing together a model of reinforcement learning
% and structure learning decision process
%
% Jun Seok Lee - Nov 2019

% Import experimental values
subj        = 1;
filename    = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',subj,subj);
load(filename,'expe');

% Initial filter and estimation variables
nu      = 0;                    % init process uncertainty
omega   = expe(1).cfg.sgen^2;   % observation noise
zeta    = 1;                    % learning noise

nsims   = 100; % number of observers to simulate
nt = numel(expe(1).blck);   % number of trials in block
nb = numel(expe);           % number of blocks in experiment

% RL variables
ests    = []; % estimations (value of estimated mean)
k       = []; % learning rates
obs     = []; % observations

% SL variables
decay   = .1;  % rate of decay of the beta distribution shape parameters
alpha   = ones(nsims,3);     % shape parameter for stay
beta    = ones(nsims,3);     % shape parameter for switch
p_rep   = zeros(nsims,1,nb); % probability that one should repeat

ests    = zeros(nsims,nt+1,nb);	% init estimation of value diffs (A-B) at 0 (flat prior)
p_a     = zeros(nsims,nt,nb); % probability that a is the good shape from reinforcement learning 
choices = zeros(nsims,nt,nb);   % simulated choices
k       = ones(nsims,nt,nb);	% initial learning rate

insts_blck = zeros(1,3);
prev_ib    = zeros(1,3);


shape_a = zeros(1,3);
shape_b = zeros(1,3);

% Simulate
for ib = 4:nb
    
    condtype = expe(ib).type;
    switch condtype % block type
        case 'rep'
            sl_index = 1;
        case 'alt'
            sl_index = 2;
        case 'rnd'
            sl_index = 3;
    end
        
    if ismember(ib, 4:6) % first encounters of blocks of each type
        % assign 1st and 2nd shape within each block type
        shape_a(sl_index) = expe(ib).shape(1);
        shape_b(sl_index) = expe(ib).shape(2);
        
        w           = ones(nsims,1).*1e6; % infinite variance  
        p_a(:,1,ib) = .5;  % probability of the good shape being shape_a
    else
        % consider the altered prior based on structure learned distribution
        
        % need to figure out how to code repeating or switching
        % if previous choice was a, then p_a = p_rep (which is 1-p_rep in practice, here)
        % if previous choice was b, then p_a = 1-p_rep (which is p_rep in practice, here)
        %{
        p_rep_mult  = eq(choices(:,nt,prev_ib(sl_index)), shape_a(sl_index).*size(nsims,1)); % compare prev choices w/ shape a
        p_a_temp    = double(p_rep_mult) - p_rep(:,1,prev_ib(sl_index));
        p_a_temp(p_a_temp<0) = p_rep(find(p_a_temp<0),1,prev_ib(sl_index));
        p_a(:,1,ib) = p_a_temp;
        %}
        p_a(:,1,ib) = .5;
        %[~,w]       = betastat(alpha(:,sl_index),beta(:,sl_index));
        w           = ones(nsims,1).*1e6; % infinite variance  
        
    end
    
    k(:,1,ib) = lrate(w, nu, omega); % set learning rate
    
    for it = 1:nt
        if it == 1
            % estimate is already set to 0
            for is = 1:nsims
                choices(is,it,ib) = datasample([shape_a(sl_index) shape_b(sl_index)], 1, 'Weights', [p_a(is,it,ib) 1-p_a(is,it,ib)]);
            end
        else
            
            % outcome
            obs = expe(ib).blck(it);
            
            % estimate
            ests(:,it+1,ib) = kalman(ests(:,it,ib),k(:,it,ib),obs,zeta);
            
            % update learning rate
            k(:,it+1,ib) = lrate(w, nu, omega);
            w = (1-k(:,it+1,ib)).*(w+nu);
            
            % bayesian update of probability that "a" is the good shape 
            p_a(:,it,ib) = 1-normcdf(0,ests(:,it+1,ib),omega);
            
            % choice based on above proba
            for is = 1:nsims
                choices(is,it,ib) = datasample([shape_a(sl_index) shape_b(sl_index)], 1, 'Weights', [p_a(is,it,ib) 1-p_a(is,it,ib)]);
            end
        end

    end % /trial loop
    
    insts_blck(sl_index) = insts_blck(sl_index) + 1;
    inst_blck = insts_blck(sl_index);
    
    % decay SL beta distrib shape parameters
    % (maybe the decay rate should be based on how strong the belief was at the end of
    %   a block???)
    for isl = 1:3
        alpha(:,isl)  = alpha(:,isl)-decay;
        beta(:,isl)   = beta(:,isl)-decay;
        
        % making sure that these shape parameters remain at least at unity
        alpha(alpha<1) = 1;
        beta(beta<1) = 1;
    end
    
    % structure learning happens at the end of blocks, starting from the end of the
    %   2nd instance of a block
    if inst_blck > 1
        % track whether actions at end of blocks were repeated or switched from the block instance before
        rep_trckr = eq(choices(:,nt,ib),choices(:,nt,prev_ib(sl_index)));
        
        alpha(:,sl_index) = alpha(:,sl_index) + double(rep_trckr);
        beta(:,sl_index) = beta(:,sl_index) + double(~rep_trckr);
    end
    % track this instance of a given block type for future comparison
    prev_ib(sl_index) = ib; 
    
    % update probability that one should NOT repeat previous choice
    for is = 1:nsims
        % the below function should be 1-betacdf, but this will be done above to
        % facilitate ease of programming 
        p_rep(is,1,ib) = betacdf(.5,alpha(is,sl_index),beta(is,sl_index));
    end
    
end % /block loop

%%

% Local functions
function out = kalman(x,k,o,zeta) %(previous estimate, kalman gain, observation)
    d1 = size(x,1);
    d2 = size(x,2);
    out = x+k.*(o-x).*(1+randn(d1,d2).*zeta);
end

function out = lrate(w,nu,omega)
    out = (w+nu)./(w+nu+omega);
end