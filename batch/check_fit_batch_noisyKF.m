% check_fit_batch_noisyKF
%
% Objective: Check fitted parameters for the noisy KF model
%
% Jun Seok Lee <jlexternal@gmail.com>

clear all
clc

% gather all out_fit structures into 1 file
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

out_fit_all = cell(3,5,nsubj);
out_fit_old = cell(3,5,nsubj);

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    
    % load file
    filename = sprintf('./param_fit_noisyKF/rep_alt/out_fit_noisyKF_rep_alt_%d_%d.mat',nsubj,isubj);
    out_new = load(filename);
    filename = sprintf('./param_fit_PF_KFpriorbias/out/out_fit_KFpriorbias_theta_biasSubj_tsu_%d_%02d.mat',nsubj,isubj);
    out_old = load(filename);
    
    for ic = 1:2
        for iq = 1:4
            out_fit_all{ic,iq,jsubj} = out_new.out_fit{ic,iq,isubj};
            out_fit_old{ic,iq,jsubj} = out_old.out_fit{ic,iq,isubj};
        end
    end
    filename = sprintf('./param_fit_noisyKF/rnd/out_fit_noisyKF_rnd_%d_%d.mat',nsubj,isubj);
    load(filename);
    out_fit_all{3,5,jsubj} = out_fit{3,5,isubj};
    
    filename = sprintf('./param_fit_PF_KFunbiased/options_fit/out_fit_KFunbiased_ths_sym_upd_28_%02d.mat',isubj);
    load(filename);
    out_fit_old{3,5,jsubj} = out_fit{3,isubj};
end
clearvars out_fit out_old out_new
%% check parameter correlation on the random condition

xmode = nan(2,nsubj,3);
xmean = nan(2,nsubj,3);
theta = nan(1,nsubj);

for isubj = 1:nsubj
    for ip = 1:3
        xmode(1,isubj,ip) = out_fit_old{3,5,isubj}.xmap(ip);
        xmode(2,isubj,ip) = out_fit_all{3,5,isubj}.xmap(ip);
        
        xmean(1,isubj,ip) = out_fit_old{3,5,isubj}.xavg(ip);
        xmean(2,isubj,ip) = out_fit_all{3,5,isubj}.xavg(ip);
    end
    theta(isubj) = out_fit_all{3,5,isubj}.xavg(4);
end

figure
for ip = 1:3
    subplot(2,3,ip)
    scatter(xmean(1,:,ip),xmean(2,:,ip));
    title(sprintf('param %d, mean',ip))
    if ip <3
        xlim([0 1])
        ylim([0 1])
    else
        xlim([0 3])
        ylim([0 3])
    end
    xlabel('old fit')
    ylabel('new fit')
    subplot(2,3,3+ip)
    scatter(xmode(1,:,ip),xmode(2,:,ip));
    title(sprintf('param %d, mode',ip))
    if ip <3
        xlim([0 1])
        ylim([0 1])
    else
        xlim([0 3])
        ylim([0 3])
    end
    xlabel('old fit')
    ylabel('new fit')
end

% compare selection noise w/ learning noise
figure
subplot(1,2,1)
scatter(xmean(2,:,3),theta)
lsline
xlabel('mean(zeta)')
subplot(1,2,2)
scatter(xmode(2,:,3),theta)
lsline
xlabel('mode(zeta)')
sgtitle('theta vs zeta (new fit)')
% compare selection noise with initial learning rate
figure
subplot(1,2,1)
scatter(xmean(2,:,1),theta)
lsline
xlabel('mean(kini)')
subplot(1,2,2)
scatter(xmode(2,:,1),theta)
lsline
xlabel('mode(kini)')
sgtitle('theta vs kini (new fit)')

%% check parameter correlation on the rep/alt condition

xmode = nan(2,nsubj,4,4,2); % old/new, subj, params, quarter, icond
xmean = nan(2,nsubj,4,4,2);

for isubj = 1:nsubj
    for ic = 1:2
        for iq = 1:4
            for ip = 1:4
                xmode(1,isubj,ip,iq,ic) = out_fit_old{ic,iq,isubj}.xmap(ip);
                xmode(2,isubj,ip,iq,ic) = out_fit_all{ic,iq,isubj}.xmap(ip);

                xmean(1,isubj,ip,iq,ic) = out_fit_old{ic,iq,isubj}.xavg(ip);
                xmean(2,isubj,ip,iq,ic) = out_fit_all{ic,iq,isubj}.xavg(ip);
            end
        end
    end
end

figure
for ip = 1:4
    for iq = 1:4
        % mean
        subplot(4,4,4*(ip-1)+iq)
        for ic = 1:2
            scatter(xmean(1,:,ip,iq,ic),xmean(2,:,ip,iq,ic),'o'); % mean
            hold on
            scatter(xmode(1,:,ip,iq,ic),xmode(2,:,ip,iq,ic),'+'); % mode
            plot([0 3],[0 3],':');
            hold off
        end
        if ip == 1
            title(sprintf('Quarter %d',iq),'FontSize',12);
        end
        if ip <3
            xlim([0 1])
            ylim([0 1])
        else
            xlim([0 3])
            ylim([0 3])
        end
        if iq == 1
            if ip == 1
                ylabel('kini (new)','FontSize',12)
            elseif ip == 2
                ylabel('kinf (new)','FontSize',12)
            elseif ip == 3
                ylabel('zeta (new)','FontSize',12)
            else
                ylabel('theta (new)','FontSize',12)
            end
        end
        if ip == 4
            xlabel('old')
        end
    end
end
%%
% compare selection noise w/ learning noise
figure
subplot(1,2,1)
scatter(reshape(xmean(2,:,3,:),[1 8*28]),reshape(xmean(2,:,4,:),[1 8*28]))
lsline
xlabel('zeta mean')
ylabel('theta mean')
subplot(1,2,2)
scatter(reshape(xmode(2,:,3,:),[1 8*28]),reshape(xmode(2,:,4,:),[1 8*28]))
lsline
xlabel('zeta mode')
ylabel('theta mode')
sgtitle('theta vs zeta on the new fit')
% compare selection noise with initial learning rate
figure
subplot(1,2,1)
scatter(reshape(xmean(2,:,1,:),[1 8*28]),reshape(xmean(2,:,4,:),[1 8*28]))
lsline
xlabel('kini mean')
ylabel('theta mean')
subplot(1,2,2)
scatter(reshape(xmode(2,:,1,:),[1 8*28]),reshape(xmode(2,:,4,:),[1 8*28]))
lsline
xlabel('kini mode')
ylabel('theta mode')
sgtitle('theta vs kini on the new fit')

%% check posterior distribution of parameters

% cfg
isubj = 28; % max 28
icond = 1;
itime = 4;

% dont touch
bounds_n = [0 1; 0 1; 0 3; 0 3]';
bounds_o = bounds_n;
if icond == 3
    itime = 5;
    bounds_o = [0 1; 0 1; 0 3]';
end

condstr = {'rep','alt','rnd'};
cornerplot(vbmc_rnd(out_fit_all{icond,itime,isubj}.vp,1e5),out_fit_all{icond,itime,isubj}.xnam,[],bounds_n);
sgtitle(sprintf('new fit, cond %s, quar %d, subj %d',condstr{icond},itime,isubj))
cornerplot(vbmc_rnd(out_fit_old{icond,itime,isubj}.vp,1e5),out_fit_old{icond,itime,isubj}.xnam,[],bounds_o);
sgtitle(sprintf('old fit, cond %s, quar %d, subj %d',condstr{icond},itime,isubj))





