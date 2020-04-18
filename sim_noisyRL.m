function [out] = sim_noisyRL(cfg)
%  SIM_NOISYRL  Simulate noisy RL model in two-armed bandit task
%
%  Usage: [out] = SIM_NOISYRL(cfg)
%
%  The configuration structure cfg should contain the following fields:
%    * rew    - the reward values associated with each option (in [0,1])
%    * trl    - the trial number in the current block
%    * fbtype - the feedback type (1:partial or 2:complete)
%    * nsim   - the number of simulations
%
%  The configuration structure cfg can additionally contain a cue field,
%  which indicates whether the trial corresponds to a cued trial in which
%  the subject was required to select a pre-cued option (cue = 1) rather
%  than choosing the option he/she wanted to sample (cue = 0). Simulated
%  responses in these cued trials are random (i.e., they follow the cue which
%  is unrelated to option values). The function assumes that the task contains
%  no cued trials if no cue field is found in the configuration structure.
%
%  The noisy RL model has four free parameters, which should be provided as
%  fields in the configuration structure:
%    * alphac - the learning rate associated with the chosen option
%    * alphau - the learning rate associated with the unchosen option
%    * zeta   - the scaling of learning noise with the prediction error
%    * ksi    - the constant term of learning noise
%    * tau    - the temperature of the softmax action selection policy
%
%  The RL model can be made to learn the policy (the difference in value
%  between option 1 and option 2) rather than the values associated with each
%  option by setting alphau to not-a-number in the configuration structure.
%
%  Reference:
%  Findling, C., Skvortsova, V., Dromnelle R., Palminteri S., and Wyart, V.
%  (2018). "Computational noise in reward-guided learning drives behavioral
%  variability in volatile environments." bioRxiv, doi:10.1101/439885.
%
%  Valentin Wyart <valentin.wyart@ens.fr>

% check required parameters
if ~all(isfield(cfg,{'rew','trl','fbtype','nsim'}))
    error('Incomplete configuration structure!');
end
if ~all(isfield(cfg,{'alphac','alphau','zeta','tau'}))
    error('Missing parameter values!');
end
% check optional parameters
if ~isfield(cfg,'cue')
    cfg.cue = zeros(size(cfg.resp));
end

% get data
rew  = cfg.rew; % rewards
trl  = cfg.trl; % trial number in current block
cue  = cfg.cue; % cued trial?

ntrl = numel(trl); % number of trials
nsim = cfg.nsim; % number of simulations

% account for feedback type (1:partial 2:complete)
fbtype = cfg.fbtype;
if fbtype == 1
    % set unchosen rewards to overall mean
    indx = sub2ind([ntrl,2],(1:ntrl)',3-resp);
    rew(indx) = 0.5;
end

% get parameter values
alphac = cfg.alphac; % learning rate of chosen option
alphau = cfg.alphau; % learning rate of unchosen option
zeta   = cfg.zeta; % scaling of learning noise with prediction error
ksi    = cfg.ksi; % constant term of learning noise
tau    = cfg.tau; % softmax temperature

% check whether policy learning
if isnan(alphau)
    alphau = alphac;
end

% check parameter values
if alphac < 0 || alphac > 1
    error('Learning rate of chosen option out-of-range!');
end
if alphau < 0 || alphau > 1
    error('Learning rate of unchosen option out-of-range!');
end
if zeta < 0 || ksi < 0
    error('Learning noise out-of-range!');
end
if tau < 0
    error('Softmax temperature out-of-range!');
end

% run simulations
r = zeros(ntrl,nsim); % response
q = zeros(ntrl,2,nsim); % Q-values
for itrl = 1:ntrl
    % 1st response of each block (random)
    if trl(itrl) == 1
        r(itrl,:) = 1+(rand(1,nsim) > 0.5);
        q(itrl,:,:) = 0.5;
        continue
    end
    % 1/ update Q-values
    q(itrl,:,:) = q(itrl-1,:,:);
    alpha = nan(1,2,nsim);
    i1 = r(itrl-1,:) == 1;
    i2 = ~i1;
    alpha(1,1,i1) = alphac; alpha(1,2,i1) = alphau;
    alpha(1,1,i2) = alphau; alpha(1,2,i2) = alphac;
    for i = 1:2
        e = rew(itrl-1,i)-q(itrl,i,:); % prediction error
        s = sqrt(zeta^2*e.^2+ksi^2);
        q(itrl,i,:) = normrnd(q(itrl,i,:)+alpha(1,i,:).*e,s);
    end
    % 2/ compute response
    if cue(itrl) == 0 % free trial
        qd = reshape(q(itrl,1,:)-q(itrl,2,:),[1,nsim]);
        r(itrl,:) = 1+(rand(1,nsim) > 1./(1+exp(-qd/tau)));
    else % cued trial (random)
        r(itrl,:) = 1+(rand(1,nsim) > 0.5);
    end
end

% create output structure
out        = [];
out.resp   = r; % simulated responses
out.rew    = cfg.rew; % rewards
out.trl    = cfg.trl; % trial number in current block
out.cue    = cfg.cue; % cued trial?
out.alphac = alphac; % learning rate of chosen option
out.alphau = alphau; % learning rate of unchosen option
out.zeta   = zeta; % scaling of learning noise with prediction error
out.ksi    = ksi; % constant term of learning noise
out.tau    = tau; % softmax temperature

end
