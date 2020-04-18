% Integration of Reinforcement Learning and Structure Learning (alpha 1)
%
% This code is a first attempt at bringing together a model of reinforcement learning
% and structure learning decision process
%
% Comments:
% This script analyzes only a single experimental file at a time, and thus is flawed
% for generating statistics about the performance of simulations on multiple
% instances of the experiment. This will be ameliorated in scratch6.m (2 Dec 2019)
%
% Jun Seok Lee - Nov 2019

% Import experimental values
subj        = 1;
filename    = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',subj,subj);
load(filename,'expe');

% Initial filter and estimation variables
nu      = 0;                    % init process uncertainty
omega   = expe(1).cfg.sgen^2;   % observation noise
zeta    = .6;                    % learning noise

nsims   = 100; % number of observers to simulate
nt = numel(expe(1).blck);   % number of trials in block
nb = numel(expe);           % number of blocks in experiment

% RL variables
ests    = []; % estimations (value of estimated mean)
k       = []; % learning rates
obs     = []; % observations

% SL variables
decay   = .4;  % rate of decay of the beta distribution shape parameters
alpha   = ones(nsims,3,nb);     % shape parameter for stay
beta    = ones(nsims,3,nb);     % shape parameter for switch
p_rep   = zeros(nsims,1,nb);    % probability that one should repeat

wgt     = flip(linspace(.1,.9,(nb-3)/3));   % amount of weight to give to RL during softmax choice

ests    = zeros(nsims,nt+1,nb);	% init estimation of value diffs (A-B) at 0 (flat prior)
p_a     = zeros(nsims,nt+1,nb); % probability that a is the good shape from reinforcement learning 
p_a_sl  = zeros(nsims,nt+1,nb); % probability that a is the good shape from structure learning 

choices     = zeros(nsims,nt,nb);   % simulated choices 
choices_sm  = zeros(nsims,nt,nb);   % simulated choices (softmax on RL and SL)
k       = ones(nsims,nt+1,nb);	% initial learning rate

insts_blck = zeros(1,3);
prev_ib    = zeros(1,3);

actionbias = false;

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
    
    insts_blck(sl_index) = insts_blck(sl_index) + 1;
    inst_blck = insts_blck(sl_index);    
        
    if ismember(ib, 4:6) % first encounters of blocks of each type
        shape_a(sl_index) = expe(ib).shape(1);
        shape_b(sl_index) = expe(ib).shape(2);
        w           = ones(nsims,1).*1e6; % infinite variance  
        p_a(:,1,ib) = .5;  % probability of the good shape being shape_a
    else
        % consider the altered prior based on structure learned distribution
        
        % need to figure out how to code repeating or switching
        % if previous choice was a, then p_a = p_rep (which is 1-p_rep in practice, here)
        % if previous choice was b, then p_a = 1-p_rep (which is p_rep in practice, here)
        
        p_rep_mult  = eq(choices(:,nt,prev_ib(sl_index)), shape_a(sl_index).*size(nsims,1)); % compare prev choices w/ shape a
        p_a_temp    = double(p_rep_mult) - p_rep(:,1,prev_ib(sl_index));
        p_a_temp(p_a_temp<0) = p_rep(find(p_a_temp<0),1,prev_ib(sl_index));
        p_a(:,1,ib) = p_a_temp;
        
        p_a_sl(:,1,ib) = p_a(:,1,ib); % 
        
        %p_a(:,1,ib) = .5;
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
            if expe(ib).shape(1) == shape_a(sl_index)
                obs = expe(ib).blck(it);
            else
                obs = -expe(ib).blck(it);
            end
            
            % estimate
            ests(:,it+1,ib) = kalman(ests(:,it,ib),k(:,it,ib),obs,zeta);
            
            % update learning rate
            k(:,it+1,ib) = lrate(w, nu, omega);
            w = (1-k(:,it+1,ib)).*(w+nu);
            
            % bayesian update of probability that "a" is the good shape 
            p_a(:,it,ib) = 1-normcdf(0,ests(:,it+1,ib),omega);
            
            % choice based on above proba - NOTE: this is not softmax
            % Note: Here the structure learned prior introduces a bias in action, but
            %       not in the value of a given shape
            if actionbias
                for is = 1:nsims
                    choices(is,it,ib) = datasample([shape_a(sl_index) shape_b(sl_index)], 1, 'Weights', [p_a(is,it,ib) 1-p_a(is,it,ib)]);
                end
            else
                s_hat = log(p_a_sl(:,1,ib)./(1-p_a_sl(:,1,ib)));
                q_hat = log(p_a(:,1,ib)./(1-p_a(:,1,ib)));
                p_a(:,1,ib) = (1+exp(-(wgt(inst_blck)*q_hat+(1-wgt(inst_blck))*s_hat))).^-1;
                for is = 1:nsims
                    choices(is,it,ib) = datasample([shape_a(sl_index) shape_b(sl_index)], 1, 'Weights', [p_a(is,it,ib) 1-p_a(is,it,ib)]);
                end
            end
        end

    end % /trial loop
    
    % decay SL beta distrib shape parameters
    % (maybe the decay rate should be based on how strong the belief was at the end of
    %   a block???)
    for isl = 1:3
        alpha_temp  = alpha(:,isl,ib-1)-decay;
        beta_temp   = beta(:,isl,ib-1)-decay;
        
        % making sure that these shape parameters remain at least at unity
        alpha_temp(alpha_temp<1)    = 1;
        beta_temp(beta_temp<1)      = 1;
        alpha(:,isl,ib) = alpha_temp;
        beta(:,isl,ib)  = beta_temp;
    end
    
    % structure learning happens at the end of blocks, starting from the end of the
    %   2nd instance of a block
    if inst_blck > 1
        % track whether actions at end of blocks were repeated or switched from the block instance before
        rep_trckr = eq(choices(:,nt,ib),choices(:,nt,prev_ib(sl_index)));
        
        alpha(:,sl_index,ib) = alpha(:,sl_index) + double(rep_trckr);
        beta(:,sl_index,ib) = beta(:,sl_index) + double(~rep_trckr);
    end
    % track this instance of a given block type for future comparison
    prev_ib(sl_index) = ib; 
    
    % update probability that one should repeat previous choice
    for is = 1:nsims
        p_rep(is,1,ib) = betacdf(.5,alpha(is,sl_index,ib),beta(is,sl_index,ib));
    end
    
end % /block loop

%% tests
for i = 1:3
    mean(alpha(:,i,nb))
    mean(beta(:,i,nb))
    [h,p] = ranksum(alpha(:,i,nb),beta(:,i,nb));
end

% plot trajectories for each condition
% plot evolution of memory on switch or stay for each condition
%% Plots
xs = 0:.01:1;
xr = 1:16;
xc = linspace(1,0.2,(nb-3)/3);

title1 = 'Dynamics of optimal observer over blocks';
if actionbias
    title2 = 'Bias on initial action selection';
else
    title2 = sprintf('Weighted softmax between RL & SL (RL wgt = %0.1f)',wgt(numel(wgt)));
end
mgen = expe(1).cfg.mgen;
clf;
figure(1);
for ib = 4:nb
    condtype = expe(ib).type;
    
    switch condtype % block type
        case 'rep'
            sl_index = 1;
            rgb = [1,xc(floor((ib-1)/3)),xc(floor((ib-1)/3))];
        case 'alt'
            sl_index = 2;
            rgb = [xc(floor((ib-1)/3)),1,xc(floor((ib-1)/3))];
        case 'rnd'
            sl_index = 3;
            rgb = [xc(floor((ib-1)/3)),xc(floor((ib-1)/3)),1];
    end
    
    subplot(3,3,3*sl_index-2);
    plot(xs,betapdf(xs,mean(alpha(:,sl_index,ib)),mean(beta(:,sl_index,ib))),'Color',rgb);
    xline(.5);
    title(sprintf('Evolution of beta distrib. (%s)',condtype));
    hold on;
    
    subplot(3,3,3*sl_index-1);
    plot(xr,mean(ests(:,2:end,ib)),'Color',rgb);
    yline(mgen);
    ylim([-.5 .5]);
    xlim([1 16]);
    title(sprintf('KF optimal tracking (%s)',condtype));
    hold on;

    subplot(3,3,3*sl_index);
    plot(xr,mean(bsxfun(@eq,choices(:,:,ib),expe(ib).shape(1)*ones(nsims,nt))),'Color',rgb);
    xlim([1 16]);
    title(sprintf('Accuracy (%s)',condtype));
    
    sgtitle(sprintf('%s \n decay param: %0.1f, learning noise: %0.1f \n %s', title1, decay, zeta, title2),'FontName','Helvetica');
    hold on;
    pause(.05);
    
end
hold off;

%%

% Local functions

% Kalman Filter
function out = kalman(x,k,o,zeta) %(previous estimate, kalman gain, observation)
    d1 = size(x,1); % number of rows
    d2 = size(x,2); % number of columns
    out = x+k.*(o-x).*(1+randn(d1,d2).*zeta);
end

% Update learning rate
function out = lrate(w,nu,omega) % (posterior uncert. ,proc. uncert., observation noise)
    out = (w+nu)./(w+nu+omega);
end

function draw_trajectories
    disp('yeet');
end

function draw_betadist
    disp('yeet');
end