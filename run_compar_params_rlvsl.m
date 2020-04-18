clear all;
clc;

%% Generate blocks
% Trial-to-trial RL difficulty calculation
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

nexp = 1; % number of quarters to simulate on

cfg         = struct;
cfg.ntrls   = 16;
cfg.mgen    = 55;
cfg.sgen    = sig_opti;
cfg.nbout   = 4*nexp; 
cfg.ngen    = 10000;
cfg.mcrit   = 5;      % max distance of sampling mean from true mean
cfg.scrit   = 10;      % max difference between sampling spread and true spread

blks = round(gen_blck_rlvsl(cfg))/100;

%%
nsmp = 1e3; % number of samples contributing to the distribution
pprec = .1; % increment value in a parameter set

% Model parameters
pnam = {}; % parameter name
pmin = []; % minimum value
pmax = []; % maximum value
pset = {};
% 1/ learning noise
pnam{1,1} = 'zeta';
pmin(1,1) = 0.001;
pmax(1,1) = 2;
pset{1} = linspace(pmin(1,1),pmax(1,1),floor((pmax(1,1)-pmin(1,1))/pprec));
% 2/ learning rate asympote
pnam{1,2} = 'alpha';
pmin(1,2) = 0;
pmax(1,2) = .5;
pset{2} = linspace(pmin(1,2),pmax(1,2),floor((pmax(1,2)-pmin(1,2))/pprec));
% 3/ structure learned prior (in units of probability)
pnam{1,3} = 'prior';
pmin(1,3) = 0;
pmax(1,3) = 1;
pset{3} = linspace(pmin(1,3),pmax(1,3),floor((pmax(1,3)-pmin(1,3))/pprec));
% Make parameter sets
psets = combvec(pset{1},pset{2}); % func. combvec from Deep Learning Toolbox
psets = combvec(psets,pset{3});


%%
ntrl = cfg.ntrls;

for iset = size(psets,2) % for each set of model parameters
    zeta    = psets(1,iset);
    alpha   = psets(2,iset);
    prior   = psets(3,iset);
    vs = (sig_opti/100)^2;    % sampling variance
    vn = (alpha/(1-alpha))^2; % process noise (calculated from value of asympote parameter)
    % run simulated data for 4 blocks of experimental data
    
    for iblk = 1:size(blks,1)
        rew = blks(iblk,:);
        
        p   = zeros(ntrl,nsmp); % response probability
        e   = zeros(ntrl,nsmp); % prediction errors
        q   = zeros(ntrl,nsmp); % filtered Q-values
        z   = zeros(ntrl,nsmp); % filtered learning errors
        qlt = zeros(ntrl,nsmp); % logit on Q-values
        slt = zeros(ntrl,nsmp); % logit on structure learned prior
        k   = zeros(ntrl,nsmp); % kalman gains
        v   = zeros(ntrl,nsmp); % covariance noise

        for itrl = 1:ntrl
            % 1st trial response
            if itrl == 1
                q(itrl,:,:) = 0.5;
                qlt(itrl,:) = 1e-6;
                slt(itrl,:) = log(prior)-log(1-prior);
                k(itrl,:) = 0;
                v(itrl,:) = 1e6;
                p(itrl,:)   = (1+exp(-(qlt(itrl,:)+slt(itrl,:)))).^-1; 
                continue
            end

            % 1/ update Q-values
            q(itrl,:) = q(itrl-1,:);
            v(itrl,:) = v(itrl-1,:);

            e(itrl,:)   = rew(itrl-1)-q(itrl,:);            % prediction error
            s           = sqrt(zeta^2*e(itrl,:).^2);        % learning noise s.d.
            k(itrl,:)   = v(itrl,:)./(v(itrl,:)+vs);    % kalman gain update
            q(itrl,:)   = q(itrl,:)+k(itrl,:).*e(itrl,:).*(1+randn(1,nsmp)*zeta);   % noisy learning
            v(itrl,:)   = (1-k(itrl,:)).*v(itrl,:)+vn;      % covariance noise update

            % 2/ compute response probability
            p(itrl,:)   = 1-normcdf(.5,q(itrl,:),s);                % probability of response from strictly RL
            qlt(itrl,:) = log(p(itrl,:))-log(1-p(itrl,:));           % logit contribution of RL
            slt(itrl,:) = log(prior)-log(1-prior);                   % logit contribution of SL 

            p(itrl,:)   = (1+exp(-(qlt(itrl,:)+slt(itrl,:)))).^-1; % integrated probability of response

        end

    end
    
    % log the trajectory to be compared later

end


% output: trajectories of correct response given one instance of stimuli (generated experiment)



% calculate KL divergence between distributions created from parameter sets

% normalize the 