
%%
clear all
close all
clc

nb    = 20; % number of blocks
nt    = 10; % number of trials per block
ns    = 1e3; % number of simulations
r_mu  = 0.1; % mean of the value difference between the 'good' and 'bad' symbols
r_sd  = 0.2; % standard deviation of the value difference between the 'good' and 'bad' symbols
ctype = 'alt'; % rule for updating which symbol is the good one across successive blocks

zeta = 1; % Kalman filtering noise for value learning
alpha = 0.2; % constant learning rate for structure learning

mp = nan(nb,1,ns); 
mt = nan(nb,nt+1,ns); % posterior mean
vt = nan(nb,nt+1,ns); % posterior variance
vs = r_sd^2; % sampling variance

switch ctype
    case 'rnd' % random across successive blocks
        ct = sign(randn(nb,1,ns));
    case 'alt' % always alternating
        ct = repmat([+1;-1],[nb/2,1]);
    case 'rep' % always the same
        ct = ones(nb,1,ns);
    case 'three' % three-block pattern
        ct = [1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 1 -1 1 1].';
end
rt = normrnd(r_mu,r_sd,[nb,nt,ns]);
rt = bsxfun(@times,rt,ct);

for ib = 1:nb
    if ib == 1
        mp(ib,1,:) = 0.5;  % what exactly is this quantity? - JL
        mt(ib,1,:) = 0;
        vt(ib,1,:) = 1e6; 
    else
        p1 = 1-normcdf(0,mt(ib-1,end,:),sqrt(vt(ib-1,end,:))); % integral under N(mt(ib-1),vt(ib-1)) from 0 to +inf (from 1 block before)
        if ib == 2
            mp(ib,1,:) = mp(ib-1,1,:); % if 2nd block propagate chance mp
        else
            p2 = 1-normcdf(0,mt(ib-2,end,:),sqrt(vt(ib-2,end,:))); % integral under N(mt(ib-2),vt(ib-2)) from 0 to +inf (from 2 blocks before)
            pr = p1.*p2+(1-p1).*(1-p2); % some kind of expectation of p(repeat)? p(A1)p(A2) + p(B1)p(B2)
            mp(ib,1,:) = mp(ib-1,1,:)+(pr-mp(ib-1,1,:)).*alpha; % updating of mp (p(repeat) maybe)
        end
        mt(ib,1,:) = mt(ib-1,end,:).*(2*mp(ib,1,:)-1); % propagating the posterior from previous block given some structure learned
        vt(ib,1,:) = vt(ib-1,end,:)+4*mp(ib,1,:).*(1-mp(ib,1,:)).*mt(ib-1,end,:).^2; 
    end
    for it = 2:nt+1
        kt = vt(ib,it-1,:)./(vt(ib,it-1,:)+vs);
        mt(ib,it,:) = mt(ib,it-1,:)+(rt(ib,it-1,:)-mt(ib,it-1,:)).*kt.*(1+randn(1,1,ns)*zeta);
        vt(ib,it,:) = (1-kt).*vt(ib,it-1,:);
    end
end

figure;
subplot(1,2,1);
hold on
xc = linspace(1,0.2,nb);
for ib = 1:nb
    rgb = [xc(ib),xc(ib),1];
    plot(mean(bsxfun(@eq,sign(mt(ib,2:end,:)),ct(ib,:,:)),3)','LineWidth',2,'Color',rgb);
    pause(.2);
end
xlim([0.5,nt+0.5]);
% ylim([0.6,0.9]);
subplot(1,2,2);
hold on
xc = linspace(1,0.2,nb);
for ib = 1:nb
    rgb = [xc(ib),xc(ib),1];
    plot(2:nt,mean(mt(ib,3:end,:).*mt(ib,2:end-1,:) > 0,3)','LineWidth',2,'Color',rgb);
end
xlim([0.5,nt+0.5]);
ylim([0.5,1]);
