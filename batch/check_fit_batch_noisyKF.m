% check_fit_batch_noisyKF
%
% Objective: Check fitted parameters for the noisy KF model on
%               conditions  rep and alt on each quarter 
%                           rnd across entire experiment
%
% Jun Seok Lee <jlexternal@gmail.com>

clear all
clc
%% Data organization
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

%% Check parameter correlation on the random condition

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

%% Check parameter correlation on the rep/alt condition

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
sgtitle('Old vs New fit parameters across quarters')

% compare selection noise w/ learning noise
figure
subplot(1,2,1)
scatter(reshape(xmean(2,:,3,:,:),[1 prod(size(xmode(2,:,4,:,:)))]),reshape(xmean(2,:,4,:,:),[1 prod(size(xmode(2,:,4,:,:)))]))
hold on
plot([0 3],[0 3],':');
lsline
hold off
xlabel('zeta mean')
ylabel('theta mean')
subplot(1,2,2)
scatter(reshape(xmode(2,:,3,:,:),[1 prod(size(xmode(2,:,4,:,:)))]),reshape(xmode(2,:,4,:,:),[1 prod(size(xmode(2,:,4,:,:)))]))
hold on
plot([0 3],[0 3],':');
lsline
xlabel('zeta mode')
ylabel('theta mode')
sgtitle('theta vs zeta on the new fit')
% compare selection noise with initial learning rate
figure
subplot(1,2,1)
scatter(reshape(xmean(2,:,1,:,:),[1 prod(size(xmode(2,:,4,:,:)))]),reshape(xmean(2,:,4,:,:),[1 prod(size(xmode(2,:,4,:,:)))]))
hold on
plot([0 3],[0 3],':');
lsline
xlabel('kini mean')
ylabel('theta mean')
subplot(1,2,2)
scatter(reshape(xmode(2,:,1,:,:),[1 prod(size(xmode(2,:,4,:,:)))]),reshape(xmode(2,:,4,:,:),[1 prod(size(xmode(2,:,4,:,:)))]))
hold on
plot([0 3],[0 3],':');
lsline
xlabel('kini mode')
ylabel('theta mode')
sgtitle('theta vs kini on the new fit')

%% Check posterior distribution of parameters

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

%% Test number of samples needed
% 100 samples is not good enough

if ~exist('out_fit_all','var')
    load('sim_noisyKF_paramfit/out_fit_all.mat')
end
% cfg
isubj = 5; % max 28
icond = 2;
itime = 3;

% dont touch
bounds_n = [0 1; 0 1; 0 3; 0 3]';
bounds_o = bounds_n;
if icond == 3
    itime = 5;
    bounds_o = [0 1; 0 1; 0 3]';
end
condstr = {'rep','alt','rnd'};

% 100,000 samples
cornerplot(vbmc_rnd(out_fit_all{icond,itime,isubj}.vp,1e5),out_fit_all{icond,itime,isubj}.xnam,[],bounds_n);
sgtitle(sprintf('new fit, cond %s, quar %d, subj %d',condstr{icond},itime,isubj))
% vs 
nsamples = 2500;
cornerplot(vbmc_rnd(out_fit_all{icond,itime,isubj}.vp,nsamples),out_fit_all{icond,itime,isubj}.xnam,[],bounds_n);
sgtitle(sprintf('new fit, cond %s, quar %d, subj %d',condstr{icond},itime,isubj))

%% Qualitative analysis via model simulation
addpath('../')
addpath('../Toolboxes/Rand')
load('subj_resp_rew_all.mat')
% experimental parameters
cfg = struct;
cfg.nt = 16;
cfg.ms = .55;   cfg.vs = .07413^2; 
cfg.sbias_cor = true;  
cfg.sbias_ini = true;
cfg.cscheme = 'ths';  cfg.lscheme = 'sym';  cfg.nscheme = 'upd';
cfg.ns      = 100; % 100 simulations take around 35 seconds per subject 
cfg.ksi     = 0;
cfg.epsi    = 0;
cfg.sameexpe = true;    % true if all sims see the same reward scheme

resp_sim = nan(4,16,cfg.ns,3,4,nsubj); % block, trial, nsims, condition, quarter, subject

issampling = false;

for isubj = 1:4%:nsubj
    isubj_abs = subjlist(isubj);
    fprintf('Simulating subject %d...\n',isubj)
    for icond = 1:3
        fprintf('  Simulating condition %d...\n',icond)
        cfg.nb = 4;
        for iq = 1:5
            if iq < 5
                cfg.nb = 4;
                blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            else
                if icond ~= 3
                    continue
                else
                    cfg.nb = 16;
                    blockrange = 1:16;
                end
            end
            cfg.compexpe    = subj_resp_rew_all(isubj_abs).rew_expe(blockrange,:,icond)/100;
            cfg.firstresp   = subj_resp_rew_all(isubj_abs).resp(blockrange,1,icond);

            if icond ~= 3
                %params_smpd = vbmc_rnd(out_fit_all{icond,iq,isubj}.vp,cfg.ns);
                params_smpd = out_fit_all{icond,iq,isubj}.xavg;
            else
                %params_smpd = vbmc_rnd(out_fit_all{icond,5,isubj}.vp,cfg.ns);
                params_smpd = out_fit_all{icond,5,isubj}.xavg;
            end
            if issampling
            cfg.kini    = params_smpd(:,1);
            cfg.kinf    = params_smpd(:,2);
            cfg.zeta    = params_smpd(:,3);
            cfg.theta   = params_smpd(:,4);
            else
                cfg.kini    = params_smpd(1)*ones(cfg.ns,1);
                cfg.kinf    = params_smpd(2)*ones(cfg.ns,1);
                cfg.zeta    = 0*params_smpd(3)*ones(cfg.ns,1);
                cfg.theta   = 0*params_smpd(4)*ones(cfg.ns,1);
            end
            sim_out = sim_noisyKF_fn(cfg);
            
            if iq < 5
                resp_sim(:,:,:,icond,iq,isubj) = sim_out.resp;
            else
                for jq = 1:4
                    blockrange = 4*(jq-1)+1:4*(jq-1)+4;
                    resp_sim(:,:,:,icond,jq,isubj) = sim_out.resp(blockrange,:,:);
                    
                    % this is producing some very strange curves....
                    % 1/ Check the simulator on some extreme parameter values to ensure
                    %       that it is functioning properly
                    % 2/ It may be that the vectorized version of the sampled
                    %       parameters are not being processed properly for it > 1
                end
            end
        end
    end
end

%% Import data from batch-ran model simulations
clear all

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

sbias = 'xmap';

for isubj = 1:nsubj
    % load file
    filename = sprintf('./sim_noisyKF_paramfit/out_%s/out_resp_sim_noisyKF_%02d_%02d.mat',sbias,nsubj,isubj);
    sim_out = load(filename);
    
    if isubj == 1
        resp_sim = nan(4,16,size(sim_out.resp_sim,3),3,4,nsubj);
    end
    % transfer data
    resp_sim(:,:,:,:,:,isubj) = sim_out.resp_sim(:,:,:,:,:,isubj);
end

load('subj_resp_rew_all')


%% plot learning curves

lcurve= nan(4,16,3,nsubj,2); % quarter, trial, condition, subject

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    for icond = 1:3
        for iq = 1:4
            % subjects
            resp = subj_resp_rew_all(isubj).resp;
            resp(resp==2)=0;
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            lcurve(iq,:,icond,jsubj,1) = mean(resp(blockrange,:,icond),1); % subj curve
            
            % simulations
            resp = resp_sim(:,:,:,icond,iq,jsubj);
            resp(resp==2) = 0;
            lcurve(iq,:,icond,jsubj,2) = mean(mean(resp(:,:,:),1),3);
        end
    end
end

% plot learning curves
figure
for ic = 1:3
    subplot(1,3,ic)
    for iq = 1:4
        errorbar(1:16,mean(lcurve(iq,:,ic,:,1),4),std(lcurve(iq,:,ic,:,1),1,4)/sqrt(nsubj),...
            'CapSize',0,'Color',graded_rgb(ic,iq),'LineWidth',1.5)
        hold on
        shadedErrorBar(1:16,mean(lcurve(iq,:,ic,:,2),4),std(lcurve(iq,:,ic,:,2),1,4)/sqrt(nsubj),...
            'lineprops',{'--','Color',graded_rgb(ic,iq),'LineWidth',1.5},'patchSaturation',.1)
        title(sprintf('Condition: %d',ic))
    end
    ylim([.4 1]);
    title(sprintf('Condition: %d',ic))
    hold off
end
sgtitle(sprintf('Learning curves\nError bars SEM\n param source: xmap'))

%% Local functions
function rgb = graded_rgb(ic,iq)
           
    red = [1.0 .92 .92; .98 .80 .80; .97 .64 .64; .96 .49 .49];
    gre = [.94 .99 .94; .85 .95 .83; .74 .91 .70; .63 .87 .58];
    blu = [.93 .94 .98; .78 .85 .94; .61 .74 .89; .44 .63 .84];
           
    rgb = cat(3,red,gre);
    rgb = cat(3,rgb,blu);
    
    rgb = rgb(iq,:,ic);
end