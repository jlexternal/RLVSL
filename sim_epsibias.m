% sim_epsibias

% simulate model with epsilon-greedy bias

% Experimental parameters
nb = 16;
nt = 16;

% Generative parameters of winning distribution
% with false negative rate of 25%
r_mu = .55;
r_sd = .07413;

% Model parameters
ns      = 1;    % Number of simulated agents to generate per given parameter
kini    = 1-eps;    % Initial Kalman gain
kinf    = 0+eps;    % Asymptotic Kalman gain
zeta    = .4;   % Learning noise scale
ksi     = 0;    % Learning noise constant
epsi    = .1;   % Blind structure choice

sbias_cor = true; % bias toward the correct structure
sbias_ini = true; % initial biased means

% Simulation parameters
ns      = 10;   % number of simulations/blocks of the set of params

%%

rew = round(normrnd(r_mu,r_sd,[nb nt ns]))/100; % rewards

% Kalman Filter variables
pt = nan(nt,nb,ns);  % response probability
mt = nan(nt,nb,ns);  % posterior mean
vt = nan(nt,ns);     % posterior variance
st = nan(ns);        % current-trial filtering noise
    
for ib = 1:nb
    for it = 1:nt
        % 1st response of each block (uninformative)
        if it == 1
            % structure bias
            if sbias_cor 
                rbias = 1; % toward correct choice
            else
                rbias 
            end
            % initialize KF means and variances
            if sbias_ini
                mt(it,ib,:) = r_mu;
            else
                mt(it,ib,:) = 0.5;
            end
            
            vt(it) = kini/(1-kini)*vs;
            % First response
            if sbias_cor
                pd = 1-normcdf(.5,mt(it,ib,:),sqrt(vt(it,ib,:)));
                pt(it,ib,:) = (1-epsi)*pd + epsi;
            else
                pt
            end
            
            
            continue
        end
        
        % Update Q-value
        vt(ib,it,:) = vt(ib,it-1,:);
        q(ib,it,:)  = q(ib,it-1,:);
        
        kt(ib,it,:)  = vt(ib,it,:)./(vt(ib,it,:)+vs);     % kalman gain update
        rpe(ib,it,:) = rew(ib,it-1,:)-q(ib,it,:);         % RPE calculation
        q(ib,it,:)   = q(ib,it,:) + kt(ib,it,:).*rpe(ib,it,:).*(1+randn(1,1,ns)*zeta); % update Q-value
        vt(ib,it,:)  = (1-kt(ib,it,:)).*vt(ib,it-1,:)+vd; % covariance noise update
        
        pq = 1-normcdf(0,q(ib,it,:),sqrt(vt(ib,it,:))); % probability of correct response based on Q-value
        qlt(ib,it,:) = log(pq)-log(1-pq);       % 'Q logit'
        slt(ib,it,:) = log(prior)-log(1-prior); % 'S logit'
        
        p(ib,it,:)   = (1+exp(-(qlt(ib,it,:)+slt(ib,it,:)))).^-1; % probability of correct response accounting for structure
        for is = 1:ns
            c1(ib,it,is)   = datasample([1 2],1,'Weights',[p(ib,it,is) 1-p(ib,it,is)]);
        end
        
        
        % outputs:
        % responses
        % rewards
        % sampling variance
        % number of pf samples
        
    end
end