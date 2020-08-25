% Integration of Reinforcement Learning and Structure Learning (alpha 2)
%
% This code is a third attempt at bringing together a model of reinforcement learning
% and structure learning decision process
%
% Note 1 : In previous versions, the probability of a switch was wrongfully attributed
% to the integral under the curve on the upper half of the support. Here it will be
% determined by the mean of the distribution.
%
% Note 2 : Here, we will do away with the weights, and the weighting will be taken
% care of by the rate of decay on the beta distribution parameters
%
% Jun Seok Lee - Dec 2019

close all;
clear all;

nb    = 16;  % number of blocks
nt    = 16;  % number of trials per block
ns    = 1e3; % number of simulations
r_mu  = 0.1; % mean of the value difference between the 'good' and 'bad' symbols
r_sd  = 0.2; % standard deviation of the value difference between the 'good' and 'bad' symbols

rl_traj_ends    = nan(nb,nt,3);
rlsl_traj_ends  = nan(nb,nt,3);
p_a             = nan(nb,3);

cset = ["rep", "alt", "rnd"];

%cset = "rep"   % for use when testing one case at a time
for ctype = cset

zeta = 0.5; % Kalman filtering noise for value learning 

mp = nan(nb,1,ns);      % subjective probability of shape A being right
mt = nan(nb,nt,ns);     % posterior mean
vt = nan(nb,nt,ns);     % posterior variance
kt = zeros(nb,nt,ns);   % kalman gain
vs = r_sd^2;            % sampling variance
vd = vs*0.0625;           % (perceived) process noise as multiplicative factor of vs

p_rl = zeros(nb,nt,ns); % p(shape A) from RL
p_sl = zeros(nb,ns);    % p(shape A) from SL

ab = nan(nb,2,ns);  % (ib, alpha/beta, is) parameters of Beta distribution
decay = .2;         % [0,1], 0 meaning full memory, 1 meaning full amnesia per encounter

s  = zeros(nb,1,ns);    % log odds from SL
q  = zeros(nb,nt,ns);   % log odds from RL

switch ctype
    case 'rnd' % random across successive blocks
        ct = repmat(sign(randn(nb,1)),[1,1,ns]);
        ic = 3;
    case 'alt' % always alternating
        ct = repmat([+1;-1],[nb/2,1,ns]);
        ic = 2;
    case 'rep' % always the same
        ct = ones(nb,1,ns);
        ic = 1;
end
rt = normrnd(r_mu,r_sd,[nb,nt,ns]);
rt = bsxfun(@times,rt,ct); % outcomes

for ib = 1:nb
    
    if ib == 1 % first block
        mp(ib,1,:)          = 0.5;  % probability of 1st choice (i.e. shape A)
        mt(ib,1,:)          = 0;    % mean tracked quantity
        vt(ib,1,:)          = 1e2;  % flat RL prior
        ab(ib,:,:)  = ones(1,2,ns); % flat SL prior (represented by a=b=1)
        p_sl(ib,:)          = 0.5;  % reflects SL parameters above, without having to do calculation
        
    else
        if ib == 2 % second block, 
            mp(ib,1,:) = 0.5;
            mt(ib,1,:) = 0; % assume no prior bias on tracked quantity
            ab(ib,:,:) = ones(1,2,ns); % flat SL prior
            p_sl(ib,:) = 0.5; % reflects SL parameters above, without having to do calculation
            
        else % consider switch/stay prior from Beta distribution
            ab_temp = nan(1,2,ns);
            
            % sign of ib-1 to ib-2 determines switch or stay
            ab_temp(1,1,:) = double(bsxfun(@eq,sign(mt(ib-2,end,:)),sign(mt(ib-1,end,:)))); % stays
            ab_temp(1,2,:) = double(~bsxfun(@eq,sign(mt(ib-2,end,:)),sign(mt(ib-1,end,:)))); % switches
            
            switchvar = ab_temp(1,2,:); % switches are 1
            switchvar(switchvar==0) = -1; % stays are -1
            
            ab(ib,1,:)  = ab(ib-1,1,:) + ab_temp(1,1,:).*(1-decay); % alpha++ -decay for stay
            ab(ib,2,:)  = ab(ib-1,2,:) + ab_temp(1,2,:).*(1-decay); % beta++ -decay  for switch
            
            % update p_sl where the stay/switch represents shape A
            signswitch = sign(mt(ib-1,end,:)).*switchvar; % sign of last choice * switchvar
            for is = 1:ns
                if sign(mt(ib-1,end,is)) == 1
                    p_sl(ib,is) = ab(ib,1,is)./(ab(ib,1,is)+ab(ib,2,is));
                else
                    p_sl(ib,is) = 1-((ab(ib,1,is))./(ab(ib,1,is)+ab(ib,2,is)));
                end
            end
            
            % update log odds; s-SL, q-RL
            s(ib,1,:) = log(p_sl(ib,:))-log(1-p_sl(ib,:));
            q(ib,1,:) = 0; % log(.5/.5) = 0;
            
            mp(ib,1,:) = 1./(1+exp(-(q(ib,1,:)+s(ib,1,:))));  % probability of shape A on 1st choice
            
            mt(ib,1,:) = 0; % assume no prior bias on tracked quantity
        end
        
        vt(ib,1,:) = 1e2;% the variance however is not infinite as with the 1st block
    end
    
    % going through the trials
    for it = 2:nt+1
        
        kt(ib,it-1,:)   = vt(ib,it-1,:)./(vt(ib,it-1,:)+vs); 
        mt(ib,it,:)     = mt(ib,it-1,:)+(rt(ib,it-1,:)-mt(ib,it-1,:)).*kt(ib,it-1,:).*(1+randn(1,1,ns)*zeta);
        vt(ib,it,:)     = (1-kt(ib,it-1,:)).*vt(ib,it-1,:);
         
        p_rl(ib,it,:)   = 1-normcdf(0,mt(ib,it,:),sqrt(vt(ib,it,:)));
        q(ib,it,:)      = log(p_rl(ib,it,:)) - log(1-p_rl(ib,it,:));
        mp(ib,it,:)     = (1+exp(-(q(ib,it,:)+s(ib,1,:)))).^-1;
        
        vt(ib,it,:)     = vt(ib,it,:) + vd;
        
    end
    
end

%%
h = figure(ic);
%axis tight manual % this ensures that getframe() returns a consistent size
%filename = sprintf('model_sim_%s.gif',cset);
hold on
for ib = 1:nb
    subplot(1,3,1);
    plot(mean(bsxfun(@eq,sign(mt(ib,1:end,:)),ct(ib,:,:)),3)','LineWidth',2,'Color',rgb3(ic,ib,nb));
    rl_traj_ends(ib,:,ic) = mean(bsxfun(@eq,sign(mt(ib,2:end,:)),ct(ib,:,:)),3);
    ylim([.5 1]);
    title(sprintf('RL trajectories of learning \n correct shape'));
    hold on;
    
    subplot(1,3,2);
    if strcmp(ctype,'alt') || strcmp(ctype,'rnd')
        if ct(ib) == 1
            plot(mean(mp(ib,1:end,:),3),'LineWidth',2,'Color',rgb3(ic,ib,nb));
            rlsl_traj_ends(ib,:,ic) = mean(mp(ib,2:end,:),3);
        else
            plot(-mean(mp(ib,1:end,:),3)+1,'LineWidth',2,'Color',rgb3(ic,ib,nb));
            rlsl_traj_ends(ib,:,ic) = -mean(mp(ib,2:end,:),3)+1;
        end
    else
        plot(mean(mp(ib,1:end,:),3),'LineWidth',2,'Color',rgb3(ic,ib,nb));
        rlsl_traj_ends(ib,:,ic) = mean(mp(ib,2:end,:),3);
    end
    ylim([.4 1]);
    title('trajectories of p(goodshape)');
    hold on;
    
    subplot(1,3,3);
    plot([1:ib],mean(p_sl(1:ib,:),2),'LineWidth',2,'Color',rgb3(ic,ib,nb));
    p_a(1:ib,ic) = mean(p_sl(1:ib,:),2);
    ylim([0 1]);
    title(sprintf('p(shape A) \n based on SL'));
    pause(.1);
    
    %{
    % Export to animated gif
    drawnow;
    % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
     % Write to the GIF File 
      if ib == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
     %}
end
hold off

end

%% Figures

%{
subplot(1,3,2);
hold on
for ib = 1:nb
    if strcmp(ctype,'alt') || strcmp(ctype,'rnd')
        if ct(ib) == 1
            plot(mean(mp(ib,1:end,:),3),'LineWidth',2,'Color',rgb3(ic,ib,nb));
            rlsl_traj_ends(ib,:,ic) = mean(mp(ib,2:end,:),3);
        else
            plot(-mean(mp(ib,1:end,:),3)+1,'LineWidth',2,'Color',rgb3(ic,ib,nb));
            rlsl_traj_ends(ib,:,ic) = -mean(mp(ib,2:end,:),3)+1;
        end
    else
        plot(mean(mp(ib,1:end,:),3),'LineWidth',2,'Color',rgb3(ic,ib,nb));
        rlsl_traj_ends(ib,:,ic) = mean(mp(ib,2:end,:),3);
    end
    pause(.1);
end
hold off;

subplot(1,3,3);
hold on
plot([1:nb],mean(p_sl,2),'LineWidth',2,'Color',rgb3(ic,ib,nb));
%errorbar([1:nb],mean(p_sl,2),std(p_sl,1,2),'Color',rgb3(ic,ib,nb));
p_a(:,ic) = mean(p_sl,2);
ylim([0 1]);
title(sprintf('p(shape A) \n based on SL'));
hold off;
%}
%%

figure(4);
for ic = 1:3
    subplot(2,1,1);
    centerpt = mean(rl_traj_ends(:,:,ic),2);
    hold on;
    plot([1:nb]+(-1+.2*ic),centerpt,'Color',rgb3(ic,15,nb));
    for ib = 1:nb
        errorbar(ib+(-1+.2*ic),centerpt(ib),abs(centerpt(ib)-rl_traj_ends(ib,1,ic)),abs(centerpt(ib)-rl_traj_ends(ib,end,ic)),'Color',rgb3(ic,ib,nb));
    end
    title(sprintf('comparison of RL trajectories given condition \n (bounds are min/max of trajectories in a block)'));
    xlabel('block number');
    hold off;
    
    subplot(2,1,2);
    hold on;
    centerpt = mean(rlsl_traj_ends(:,:,ic),2);
    plot([1:nb]+(-1+.2*ic),centerpt,'Color',rgb3(ic,15,nb));
    for ib = 1:nb
        errorbar(ib+(-1+.2*ic),centerpt(ib),abs(centerpt(ib)-rlsl_traj_ends(ib,1,ic)),abs(centerpt(ib)-rlsl_traj_ends(ib,end,ic)),'Color',rgb3(ic,ib,nb));
    end
    xlabel('block number');
    title(sprintf('comparison of trajectories of p(goodshape) given condition \n (bounds are min/max of trajectories in a block)'));
    hold off;
end

%% Learning rate dynamics
figure(5);
plot(1:size(kt,2),kt(1,:,1),'LineWidth',2);
hold on;
%title('Learning rate dynamics');
xlabel('trial');
ylabel('learning rate');
ylim([0,1]);
%% test

p = 0:.01:1;
sm = 1./(1+exp(-50*(p - (1-p))));
plot(p,sm);
xline(.5);
ylim([0 1])

%% functions

function colors = rgb3(ic,ib,nb)
xc = linspace(.8,.2,nb);

colors =  [1,xc(ib),xc(ib); ...
           xc(ib),1,xc(ib); ...
           xc(ib),xc(ib),1];

colors = colors(ic,:);
end





