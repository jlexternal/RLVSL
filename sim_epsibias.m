% sim_epsibias

% simulate model with epsilon-greedy bias

% Experimental parameters
nb = 16;
nt = 16;

% Generative parameters of winning distribution
% with false negative rate of 25%
r_mu = .55;
r_sd = .07413; 
vs   = r_sd^2; 

% reparameterization functions
fk = @(v)1./(1+exp(+0.4486-log2(v/vs)*0.6282)).^0.5057;
fv = @(k)fzero(@(v)fk(v)-min(max(k,0.001),0.999),vs.*2.^[-30,+30]);

% Model parameters
ns      = 1;       % Number of simulated agents to generate per given parameter % do not change
kini    = 1.0-eps;  % Initial Kalman gain
kinf    = 0.4+eps;  % Asymptotic Kalman gain
zeta    = 0.4+eps;  % Learning noise scale
ksi     = 0.0;      % Learning noise constant
epsis   = 0.0;      % Blind structure choice 0: all RL, 1: all SL

sbias_cor = true;   % bias toward the correct structure
sbias_ini = true;  % initial biased means

% Simulation settings
nexp = 10; % number of different reward schemes to try per given parameter set

sim_struct = struct;

%% Run simulation
%epsis = [0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1];
legtxt = {};

out_ctr = 0;
rt_plot = [];
for iexps = 1:nexp
    for epsi = epsis
        out_ctr = out_ctr + 1;
        % Kalman Filter variables
        pt = nan(nb,nt,ns);     % response probability
        rt = nan(nb,nt,ns);     % actual responses
        mt = nan(nb,nt,2,ns);   % posterior mean
        vt = nan(nb,nt,2,ns);   % posterior variance
        st = nan(nb,nt,2,ns);   % current-trial filtering noise
        % Reward scheme from same generator as experiment
        cfg_gb = struct; cfg_gb.ntrls = nt; cfg_gb.mgen = r_mu; cfg_gb.sgen = r_sd; cfg_gb.nbout = nb;

        rew = [];
        rew = cat(3,rew,gen_blck_rlvsl(cfg_gb));
        rew = cat(3,rew,repmat(rew,[1 1 ns-1]));

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
                        mt(ib,it,rbias,:)   = r_mu;
                        mt(ib,it,3-rbias,:) = 1-r_mu;
                    else
                        % initial tracking mean unbiased
                        mt(ib,it,:,:) = .5;
                    end
                    vt(ib,it,:) = kini/(1-kini)*vs;
                    % first trial response probability
                    if sbias_cor
                        for is = 1:ns
                            md = mt(ib,it,1,is)-mt(ib,it,2,is);
                            sd = sqrt(sum(vt(ib,it,:,is)));
                            pd = 1-normcdf(0,md,sd);
                            pt(ib,it,is) = (1-epsi)*pd + epsi;
                        end
                    else
                        pt(ib,it,:) = rbias == 1;
                    end

                    % first trial response % 1/correct, 2/incorrect
                    rt(ib,it,:) = round(pt(ib,it,:)); % argmax choice
                    rt(rt==0) = 2;
                    continue;
                end
                % update Kalman gain
                kt = reshape(vt(ib,it-1,:,:)./(vt(ib,it-1,:,:)+vs),[2 ns]);
                % update posterior mean & variance
                % chosen/unchosen (1/2) index
                c = rt(ib,it-1,:);
                u = 3-c;
                for is = 1:ns
        %            fprintf('Sim: %d\n',is);  
                    rew_seen(c(is)) = rew(ib,it-1,is); % seen reward
                    rew_seen(u(is)) = 1-rew(ib,it-1,is); % unseen reward
                    if rt(ib,it-1,is) == 2
                        rew_seen = 1-rew_seen;
                    end
                    % 1/ chosen option
                    mt(ib,it,c(is),is) = mt(ib,it-1,c(is),is) + kt(c(is)).*(rew_seen(c(is))-mt(ib,it-1,c(is),is));
                    vt(ib,it,c(is),is) = (1-kt(c(is),is)).*vt(ib,it-1,c(is),is);
                    st(ib,it,c(is),is) = sqrt(zeta^2*((rew_seen(c(is))-mt(ib,it-1,c(is),is)).^2+ksi^2)); % noise distribution (scales with RPE)
                    % 2/ unchosen option
                    mt(ib,it,u(is),is) = mt(ib,it-1,u(is),is) + kt(u(is)).*(rew_seen(u(is))-mt(ib,it-1,u(is),is));
                    vt(ib,it,u(is),is) = (1-kt(u(is),is)).*vt(ib,it-1,u(is),is);
                    st(ib,it,u(is),is) = sqrt(zeta^2*((rew_seen(u(is))-mt(ib,it-1,u(is),is)).^2+ksi^2)); % noise distribution (scales with RPE)
                end

                % variance extrapolation + diffusion process 
                vt(ib,it,:,:)  = vt(ib,it,:,:)+fv(kinf); % covariance noise update    

                % selection noise
                ssel = 0;

                % decision variable stats
                md = reshape(mt(ib,it,1,:)-mt(ib,it,2,:),[1 ns]);
                sd = reshape(sqrt(sum(st(ib,it,:,:).^2,3)+ssel^2),[1 ns]);

                % sample trial types (based on epsi param)
                isl = rand(1,ns) < epsi;
                irl = ~isl;

                % Structure-utilising agents
                if nnz(isl) > 0
                    pt(ib,it,isl) = rbias(isl) == 1;
                end
                % RL-utilising agents
                if nnz(irl) > 0
                    pt(ib,it,irl) = 1-normcdf(0,md(irl),sd(irl));
                end

                % extract choice from probabilities
                rt(ib,it,:) = round(pt(ib,it,:));
                rt(rt==0) = 2;

                % resampling w/o conditioning on response
                mt(ib,it,:,isl) = normrnd(mt(ib,it,:,isl),st(ib,it,:,isl));
                % resampling conditioning on response
                mt(ib,it,:,irl) = resample(reshape(mt(ib,it,:,irl),[2,nnz(irl)]),...
                                           reshape(st(ib,it,:,irl),[2,nnz(irl)]),...
                                           ssel,reshape(rt(ib,it,irl),[1 numel(irl(irl==1))]));
            end

            % debug code for visuals
            if false
            rt_test = rt;
            rt_test(rt_test == 2) = 0;
            shadedErrorBar(1:nt,mean(mean(rt_test(:,:,:),1,'omitnan'),3),std(mean(rt_test(:,:,:),1,'omitnan'),1,3)/sqrt(ns),...
                        'lineprops',{'LineWidth',2+epsi},'patchSaturation',.1);
            end
        end

        sim_struct(out_ctr).epsi = epsi;
        sim_struct(out_ctr).resp = rt;
        sim_struct(out_ctr).rews = rew;
        sim_struct(out_ctr).vs   = vs;
    end
    
    rt(rt==2)=0;
    rt_plot(out_ctr,:) = mean(rt,1);
    % plotting for debug
    if false
        rt(rt==2)=0;
        figure(1);
        hold on;
        shadedErrorBar(1:nt,mean(mean(rt(:,:,:),1),3),std(rt(:,:,:),1),...
                        'lineprops',{'LineWidth',2+epsi},'patchSaturation',.1);
        ylim([0 1]);
        yline(.5,'--','HandleVisibility','off');
        xticks([1:4]*4);
        xlabel('trial number');
        ylabel('proportion correct');
        title(sprintf('Params: kini:%0.2f, kinf: %0.2f, zeta: %0.2f\n nsims:%d',kini,kinf,zeta,ns))
        legtxt = [legtxt; ['epsi: ' num2str(epsi)]]; 
        ylim([.2 1]);
        legend(legtxt,'Location','southwest');
    end
end

%% plotting
figure(1);
hold on;
shadedErrorBar(1:nt,mean(rt_plot(:,:),1),std(rt_plot(:,:),1)/sqrt(nexp),...
                'lineprops',{'LineWidth',2+epsi},'patchSaturation',.1);
ylim([0 1]);
yline(.5,'--','HandleVisibility','off');
xticks([1:4]*4);
xlabel('trial number');
ylabel('proportion correct');
title(sprintf('Params: kini:%0.2f, kinf: %0.2f, zeta: %0.2f\n nsims:%d',kini,kinf,zeta,nexp))
legtxt = [legtxt; ['epsi: ' num2str(epsi)]]; 
%end

ylim([.2 1]);
legend(legtxt,'Location','southwest');

%% Recover models

% organize simulated output data
%   responses
%   rewards
%   generative sampling variance
%   number of samples for pf
%   

rec_ctr = 0;
for ipar = 1:length(sim_struct)
    rec_ctr = rec_ctr + 1;
    
    epsi = sim_struct(ipar).epsi;
    
    cfg = [];
    cfg.vs = sim_struct(ipar).vs;
    cfg.nsmp = 1e3;
    cfg.lstruct = 'sym';
    cfg.verbose = true;
    cfg.ksi = 0;
    
    for isim = 1:ns
        resp = sim_struct(rec_ctr).resp(:,:,isim); % responses
        rt   = sim_struct(rec_ctr).rews(:,:,isim); % rewards
        
    end
    
end

%% Local functions
function [xt] = resample(m,s,ssel,r)
% 1/ resample (x1-x2)
md = m(1,:)-m(2,:);
sd = sqrt(sum(s.^2,1));
td = tnormrnd(md,sqrt(sd.^2+ssel.^2),r); 
xd = normrnd( ...
    (ssel.^2.*md+sd.^2.*td)./(ssel.^2+sd.^2), ...
    sqrt(ssel.^2.*sd.^2./(ssel.^2+sd.^2)));
% 2/ resample x1 from (x1-x2)
ax = s(1,:).^2./sd.^2;
mx = m(1,:)-ax.*md;
sx = sqrt(s(1,:).^2-ax.^2.*sd.^2);
x1 = ax.*xd+normrnd(mx,sx);
% 3/ return x1 and x2 = x1-(x1-x2)
xt = cat(1,x1,x1-xd);
end

function [x] = tnormrnd(m,s,d)
% sample from truncated normal distribution
if d == 1
    x = +rpnormv(+m,s);
else
    x = -rpnormv(-m,s);
end
end
