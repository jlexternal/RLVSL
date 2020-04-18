% Integration of Reinforcement Learning and Structure Learning (alpha 2)
%
% This code is a second attempt at bringing together a model of reinforcement learning
% and structure learning decision process
%
% Jun Seok Lee - Dec 2019

nb    = 16; % number of blocks
nt    = 16; % number of trials per block
ns    = 1e3; % number of simulations
r_mu  = 0.1; % mean of the value difference between the 'good' and 'bad' symbols
r_sd  = 0.2; % standard deviation of the value difference between the 'good' and 'bad' symbols

ctype = 'rep'; % rule for updating which symbol is the good one across successive blocks

zeta = 1; % Kalman filtering noise for value learning 

mp = nan(nb,1,ns); 
mt = nan(nb,nt,ns); % posterior mean
vt = nan(nb,nt,ns); % posterior variance
kt = zeros(nb,nt,ns); % kalman gain
vs = r_sd^2; % sampling variance

p_rl = nan(nb,nt,ns); 
p_sl = nan(nb,ns);

w = .5; % weight given to RL (1-w : weight given to SL)

ab = nan(nb,2,ns); % (ib, alpha/beta, is) parameters of Beta distribution
s  = zeros(nb,1,ns); % log odds from SL
q  = zeros(nb,nt,ns); % log odds from RL
cb = [ones(ns,1) -ones(ns,1)]; % choice A or not choice A

switch ctype
    case 'rnd' % random across successive blocks
        ct = sign(randn(nb,1,ns));
    case 'alt' % always alternating
        ct = repmat([+1;-1],[nb/2,1]);
    case 'rep' % always the same
        ct = ones(nb,1,ns);
end
rt = normrnd(r_mu,r_sd,[nb,nt,ns]);
rt = bsxfun(@times,rt,ct); % outcomes

%% 

for ib = 1:nb
    
    if ib == 1 % first block
        mp(ib,1,:)          = 0.5;  % probability of 1st choice (i.e. shape A)
        mt(ib,1,:)          = 0;    % mean tracked quantity
        vt(ib,1,:)          = 1e2;  % flat RL prior
        ab(ib,:,:)  = ones(1,2,ns); % flat SL prior (represented by a=b=1)
        
    else
        if ib == 2 % second block, 
            mp(ib,1,:) = 0.5;
            ab(ib,:,:) = ones(1,2,ns); % flat SL prior
            mt(ib,1,:) = 0; % assume no prior bias on tracked quantity
            
        else % consider switch/stay prior from Beta distribution
            ab_temp = nan(1,2,ns);
            
            % sign of ib-1 to ib-2 determines switch or stay
            ab_temp(1,1,:) = double(bsxfun(@eq,sign(mt(ib-2,end,:)),sign(mt(ib-1,end,:))));
            ab_temp(1,2,:) = double(~bsxfun(@eq,sign(mt(ib-2,end,:)),sign(mt(ib-1,end,:))));
            
            ab(ib,1,:) = ab(ib-1,1,:) + ab_temp(1,1,:); % alpha++ for stay
            ab(ib,2,:) = ab(ib-1,2,:) + ab_temp(1,2,:); % beta++  for switch
            
            % update p_sl where the stay/switch represents shape A
            ind_stay = find(ab_temp(1,1,:)==1);
            ind_swit = find(ab_temp(1,2,:)==1);
            p_sl(ib,ind_stay) = 1-betacdf(0.5,ab(ib,1,ind_stay),ab(ib,2,ind_stay)); % stay
            p_sl(ib,ind_swit) = betacdf(0.5,ab(ib,1,ind_swit),ab(ib,2,ind_swit));   % switch
            
            % update log odds
            s(ib,1,:) = log(p_sl(ib,:))-log(1-p_sl(ib,:));
            
            q(ib,1,:) = 0; % log(.5/.5) = 0;
            
            mp(ib,1,:) = (1+exp(-(w*q(ib,1,:)+(1-w)*s(ib,1,:)))).^-1;  % probability of 1st choice
            mt(ib,1,:) = 0; % assume no prior bias on tracked quantity
        end
        
        vt(ib,1,:) = vt(ib-1,end,:) + 4*mp(ib,1,:).*(1-mp(ib,1,:)).*mt(ib-1,end,:).^2;% the variance however is not infinite as with the 1st block
    end
    
    
    % going through the trials
    for it = 2:nt+1
        kt(ib,it-1,:)   = vt(ib,it-1,:)./(vt(ib,it-1,:)+vs);
        mt(ib,it,:)     = mt(ib,it-1,:)+(rt(ib,it-1,:)-mt(ib,it-1,:)).*kt(ib,it-1,:).*(1+randn(1,1,ns)*zeta);
        vt(ib,it,:)     = (1-kt(ib,it-1,:)).*vt(ib,it-1,:);
        
        
        p_rl(ib,it,:) = 1-normcdf(0,mt(ib,it,:),sqrt(vt(ib,it,:))); 
        
        q(ib,it,:) = log(p_rl(ib,it,:)) - log(1-p_rl(ib,it,:));
        
        mp(ib,it,:) = (1+exp(-(w*q(ib,it,:)+(1-w)*s(ib,1,:)))).^-1;
        
    end
    
end
%%

figure(1);
hold on
xc = linspace(1,0.2,nb);
for ib = 1:nb
    rgb = [xc(ib),xc(ib),1];
    plot(mean(bsxfun(@eq,sign(mt(ib,2:end,:)),ct(ib,:,:)),3)','LineWidth',2,'Color',rgb);
    pause(.1);
end
hold off

figure(2);
hold on
for ib = 1:nb
    rgb = [xc(ib),xc(ib),1];
    plot(mean(p_rl(ib,:,:),3)','LineWidth',2,'Color',rgb);
    pause(.1);
end
hold off;

figure(3);
hold on
for ib = 1:nb
    rgb = [xc(ib),xc(ib),1];
    plot(mean(p_sl(ib,:),2)','LineWidth',2,'Color',rgb);
    pause(.1);
end
hold off;
