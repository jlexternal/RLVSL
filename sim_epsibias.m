% sim_epsibias

% simulate model with epsilon-greedy bias

% Experimental parameters
nb = 6;
nt = 16;

% reparameterization functions
fk = @(v)1./(1+exp(+0.4486-log2(v/vs)*0.6282)).^0.5057;
fv = @(k)fzero(@(v)fk(v)-min(max(k,0.001),0.999),vs.*2.^[-30,+30]);

% Generative parameters of winning distribution
% with false negative rate of 25%
r_mu = .55;
r_sd = .07413; 
vs   = r_sd^2; 

% Model parameters
ns      = 10;       % Number of simulated agents to generate per given parameter
kini    = 1.0-eps;  % Initial Kalman gain
kinf    = 0.0+eps;  % Asymptotic Kalman gain
zeta    = 0.6;      % Learning noise scale
ksi     = 0.0;      % Learning noise constant
epsi    = 0.6;      % Blind structure choice

sbias_cor = true;   % bias toward the correct structure
sbias_ini = true;  % initial biased means


%%
% Reward scheme from same generator as experiment
cfg_gb = struct; cfg_gb.ntrls = nt; cfg_gb.mgen = r_mu; cfg_gb.sgen = r_sd; cfg_gb.nbout = nb;

rew = [];
for is = 1:ns
    rew = cat(3,rew,gen_blck_rlvsl(cfg_gb));
end

legtxt = {};
for epsi = [0 .2 .4 .6 .8 1]
% Kalman Filter variables
pt = nan(nb,nt,ns);  % response probability
rt = nan(nb,nt,ns);  % actual responses
mt = nan(nb,nt,ns);  % posterior mean
vt = nan(nb,nt,ns);  % posterior variance
st = nan(nb,nt,ns);  % current-trial filtering noise
    
rbias = nan(1,1,ns);
for ib = 1:nb
    for it = 1:nt
        if it == 1
            % determine structure bias
            if sbias_cor
                rbias(:) = 1;
            else 
                % sims are randomly choosing the 1st shape
                if ib == 1
                    rbias = reshape(randi(2,1,ns),size(rbias));
                end
            end
            % initialize KF means and variances
            if sbias_ini
                % initial tracking mean biased toward generative mean
                mt(ib,it,:) = r_mu;
            else
                % initial tracking mean unbiased
                mt(ib,it,:) = .5;
            end
            vt(ib,it,:) = kini/(1-kini)*vs;
            % first trial response probability
            if sbias_cor
                pd = 1-normcdf(.5,mt(ib,it,:),vt(ib,it,:)); % draw from initial bias
                pt(ib,it,:) = (1-epsi)*pd + epsi;
            else
                pt(ib,it,:) = rbias == 1;
            end
            
            % first trial response % 1/correct, 2/incorrect
            rt(ib,it,:) = round(pt(ib,it,:)); % argmax choice
            rt(rt==0) = 2;
            continue;
        end
        % update Kalman gain
        kt = vt(ib,it-1,:)./(vt(ib,it-1,:)+vs);
        
        % update posterior mean & variance (exact learning)
        mt(ib,it,:) = mt(ib,it-1,:) + kt.*(rew(ib,it-1,:)-mt(ib,it-1,:));
        vt(ib,it,:) = (1-kt).*vt(ib,it-1,:);
        
        % noise distribution (scales with RPE)
        st(ib,it,:) = sqrt(zeta^2*(rew(ib,it-1,:).^2+ksi^2));
        
        % variance extrapolation + diffusion process 
        vt(ib,it,:)  = vt(ib,it,:)+fv(kinf); % covariance noise update
        
        % choice selection
        ssel = 0;
        
        % sample trial types (based on epsi param)
        isl = rand(1,ns) < epsi;
        irl = ~isl;
        
        % Structure-utilising agents
        if nnz(isl) > 0
            pt(ib,it,isl) = rbias(isl) == 1;
        end
        % RL-utilising agents
        if nnz(irl) > 0
            pt(ib,it,irl) = 1-normcdf(.5,mt(ib,it,irl),sqrt(st(ib,it,irl).^2+ssel^2));
        end
        
        rt(ib,it,:) = round(pt(ib,it,:));
        rt(rt==0) = 2;
        
        % outputs:
        %   responses
        %   rewards
        %   sampling variance
        %   number of pf samples
    end
end

% plotting
rt(rt==2)=0;
figure(1);
hold on;
shadedErrorBar(1:nt,mean(mean(rt,1),3),std(mean(rt,1),1,3)/sqrt(ns)...
                ,'lineprops',{'LineWidth',2+epsi},'patchSaturation',.1);
ylim([0 1]);
yline(.5,'--','HandleVisibility','off');
xticks([1:4]*4);
xlabel('trial number');
ylabel('proportion correct');
title(sprintf('Params: kini:%0.2f, kinf: %0.2f, zeta: %0.2f',kini,kinf,zeta))
legtxt = [legtxt; ['epsi: ' num2str(epsi)]]; 
end
ylim([.4 1]);
legend(legtxt);