% scratch11.m
% Testing simulation of the simple RL model around epiphanies

% Model 1 - Standard RL, No epiphany
% RL model that does not reset from block to block 
% Q(t) = Q(t-1) + lr*(RPE)

lr = .4; % learning rate

% Model 2 - Standard RL, Epiphany/structure hard-coded 
% RL model that assumes for granted the epiphany after a certain point; structure is
%   hard-coded 
% Q(it) = Q(it-1) + lr*(RPE)
% Q(0,ib) = Q(nt,ib-1) after epiphany

% Model 3 - Dampened RPE RL, Epiphany/structure hard-coded
% RL model that assumes for granted the epiphany after a certain point; structure is
%   hard-coded; sensitivity toward RPE is dampened by a constant factor
% Q(it) = Q(it-1) + lr*(df*RPE)
% Q(0,ib) = Q(nt,ib-1) after epiphany

df = .6; % dampening factor

% Model 4 - Learning noise RL, Epiphany/structure hard-coded
% RL model that assumes for granted the epiphany after a certain point; structure is
%   hard-coded; learning noise increased after epiphany
% Q(it) = Q(it-1) + lr*(RPE)*(1+randn*zeta)
% Q(0,ib) = Q(nt,ib-1) after epiphany

zeta = .4; % learning noise factor

% Experimental parameters
nb = 16;
nt = 16;
ns = 30;

% Generative parameters of winning distribution
% with false negative rate of 25%
r_mu = 5;
r_sd = 7.413;

rew = round(normrnd(r_mu,r_sd,[nb nt ns]));

q   = nan([nb nt ns 4]); % store output: Q-values
rpe = nan(size(q));         % store output: RPEs

 
isepiph = false;
for ib = 1:nb
    if ib > nb/2
        isepiph = true;
    end

    it = 1;
    % Initialize values for all models
    if ~isepiph 
        % first half of experiment (no epiphany)
        for im = 1:4
            rpe(ib,it,:,im) = rew(ib,it,:); % random first choice
            
            q(ib,it,:,im)   = lr*rpe(ib,it,:,im);
        end
    else
        % after epiphany
        for im = 1:4
            rpe(ib,it,:,im) = rpe_model(rew(ib,it,:),im,df,zeta,ns); % correct bandit first choice
            if im == 1
                q(ib,it,:,im) = lr*rpe(ib,it,:,im);
            else
                q(ib,it,:,im) = q(ib-1,end,:,im) + lr*rpe(ib,it,:,im); % Q-values from end previous block propagated
                
            end
        end
    end

    % rest of the trials
    for it = 2:nt
        if ~isepiph 
            for im = 1:4
                rpe(ib,it,:,im)  = rew(ib,it,:)-q(ib,it-1,:,im);
                q(ib,it,:,im)    = q(ib,it-1,:,im) + lr*rpe(ib,it,:,im);
            end
        else
            for im = 1:4
                rpe(ib,it,:,im)  = rpe_model(rew(ib,it,:)-q(ib,it-1,:,im),im,df,zeta,ns);
                q(ib,it,:,im)    = q(ib,it-1,:,im) + lr*rpe(ib,it,:,im);
            end
        end
    end
end

%% Organize: Learning curves

resp = nan(ns,nt,2,4);
for im = 1:4
    
    resp_temp = sign(q(1:nb/2,:,:,im)+eps);
    resp_temp(resp_temp == -1) = 0;
    resp(:,:,1,im) = permute(squeeze(mean(resp_temp,1)),[2 1]);
    
    resp_temp = sign(q(nb/2+1:end,:,:,im)+eps);
    resp_temp(resp_temp == -1) = 0;
    resp(:,:,2,im) = permute(squeeze(mean(resp_temp,1)),[2 1]);
end

%% Plot: Learning curves
figure;
for im = 1:4
    subplot(1,2,1); % pre-epiphany
    title('Pre-epiphany');
    ylim([.75 1.05]);
    yline(1,':','HandleVisibility','off');
    if im ~= 2
        shadedErrorBar(1:nt,mean(resp(:,:,1,im)),std(resp(:,:,1,im))/sqrt(ns),'lineprops',{'LineWidth',2});
    else
        shadedErrorBar(1:nt,mean(resp(:,:,1,im)),std(resp(:,:,1,im))/sqrt(ns),'lineprops',{'-.','LineWidth',2});
    end
    xlabel('Trial number');
    ylabel('Accuracy');
    hold on;
    
    subplot(1,2,2); % post-epiphany
    title('Post-epiphany');
    ylim([.75 1.05]);
    yline(1,':','HandleVisibility','off');
    shadedErrorBar(1:nt,mean(resp(:,:,2,im)),std(resp(:,:,1,im))/sqrt(ns),'lineprops',{'LineWidth',2});
    hold on;
end
legend({'RL','RL: Epiphany','RL: Dampened RPE','RL: Learning noise'},'Location','southeast');

%% Plot: Correlation RPE to Reward

figure;
rew_pre = rew(1:nb/2,:,:);
rew_pre = rew_pre(:);
rew_post= rew(nb/2+1:end,:,:);
rew_post= rew_post(:);
for im = 1:4
    colororder = get(gca,'ColorOrder');
    subplot(1,2,1); % pre-epiphany
    title('Pre-epiphany');
    xline(0,'HandleVisibility','off');
    yline(0,'HandleVisibility','off');
    rpe_vec = rpe(1:nb/2,:,:,im);
    rpe_vec = rpe_vec(:);
    scatter(rew_pre,rpe_vec,'.','MarkerEdgeColor',colororder(im,:));
    ylabel('RPE');
    xlabel('reward');
    
    hold on;
    
    subplot(1,2,2); % post-epiphany
    title(sprintf('Post-epiphany\nLearning rate: %.1f \nRPE Scaling factor: %.1f \nLearning noise: %.1f',lr,df,zeta));
    xline(0,'HandleVisibility','off');
    yline(0,'HandleVisibility','off');
    rpe_vec = rpe(nb/2+1:end,:,:,im);
    rpe_vec = rpe_vec(:);
    scatter(rew_post,rpe_vec,'.','MarkerEdgeColor',colororder(im,:),'HandleVisibility','off');
    coefs = polyfit(rew_post,rpe_vec,1);
    plot([-30 40], coefs(1)*[-30 40]+coefs(2),'LineWidth',1.2,'Color',colororder(im,:));
    regress(rpe_vec,rew_post)
    hold on;
end
legend({'RL','RL: Epiphany','RL: Dampened RPE','RL: Learning noise'},'Location','southeast');

%% local functions
function td = rpe_model(rpe,im,df,zeta,ns)
    if ismember(im,1:2)
        td = rpe;
    elseif im == 3
        td = rpe.*df;
    elseif im == 4
        td = rpe.*reshape(1+randn(1,1,ns)*zeta,size(rpe));
    else
        error('Model index not recognized!');
    end
end