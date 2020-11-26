% check_fit_batch_noisyKF_empPrior

clear all
clc
addpath('..');
%% Data organization
% gather all out_fit structures into 1 file
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

out_fit_all = cell(3,5,nsubj);
params      = nan(3,5,nsubj,4); % cond,time,subj,param

oldpar_struct = load('out_fit_noisyKF');
pars_old = nan(3,5,nsubj,4); % not empirical prior fitted parameters

bounds = [0 0; 0 0; 0 5; 0 10];

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    
    % load new fit file
    filename = sprintf('./param_fit_PF_KFpriorbias_empPrior/out_5/out_fit_KFempPrior_tsu_%d_%02d.mat',nsubj,isubj);
    load(filename);
    
    for ic = 1:3
        if ic ~= 3
            for iq = 1:4
                out_fit_all{ic,iq,jsubj} = out_fit{ic,iq,isubj};
                for ip = 1:4
                    if ip < 3
                        params(ic,iq,jsubj,ip) = 1/(1+exp(-out_fit{ic,iq,isubj}.xmap(ip)));
                    else
                        params(ic,iq,jsubj,ip) = bounds(ip,2)./(exp(-out_fit{ic,iq,isubj}.xmap(ip))+1);
                    end
                    pars_old(ic,iq,jsubj,ip) = oldpar_struct.out_fit_all{ic,iq,jsubj}.xmap(ip);
                end
            end
        else
            out_fit_all{ic,5,jsubj} = out_fit{ic,5,isubj};
            for ip = 1:4
                if ip < 3
                    params(ic,5,jsubj,ip) = 1/(1+exp(-out_fit{ic,5,isubj}.xmap(ip)));
                else
                    params(ic,5,jsubj,ip) = exp(out_fit{ic,5,isubj}.xmap(ip));
                end
                pars_old(ic,5,jsubj,ip) = oldpar_struct.out_fit_all{ic,5,jsubj}.xmap(ip);
            end
        end
    end
end
clearvars out_fit


%% plot parameter evolution over time
parstr = {'kini','kinf','zeta','theta'};
figure
hold on
for ip = 1:4
    subplot(4,1,ip);
    for ic = 1:2
        errorbar(1:4,mean(params(ic,1:4,:,ip),3),std(params(ic,1:4,:,ip),1,3),...
                        'o','Color',graded_rgb(ic,4),'LineWidth',2,'LineStyle','none');
        hold on
        errorbar([1:4]-.1,mean(pars_old(ic,1:4,:,ip),3),std(pars_old(ic,1:4,:,ip),1,3),...
                        'x','Color',graded_rgb(ic,2),'LineWidth',1,'LineStyle','none');
    end
    ic = 3;
    shadedErrorBar([1 4],mean(params(3,5,:,ip),3)*ones(1,2),std(params(3,5,:,ip),1,3)*ones(1,2),...
                    'lineprops',{'Color',graded_rgb(ic,4)},'patchSaturation',.3);
    shadedErrorBar([1 4]-.1,mean(pars_old(3,5,:,ip),3)*ones(1,2),std(pars_old(3,5,:,ip),1,3)*ones(1,2),...
                    'lineprops',{':','Color',graded_rgb(ic,2)},'patchSaturation',.3);
    xticks(1:4)
    xticklabels(1:4)
    xlim([0.5 4.5]);
    title(sprintf(parstr{ip}));
end


%% check certain correlations between parameters

parchoice = 'prior';
ipar1 = 1;
ipar2 = 4;

if strcmpi(parchoice,'prior')
    pars = pars_old;
    titletxt = {'Fits from uniform priors'};
else
    pars = params;
    titletxt = {'Fits from empirical priors'};
end

partxt = {'kini','kinf','zeta','theta'};

par1 = [];
par2 = [];
for icond = 1:3
    if ~ismember(icond,1:2)
        par1 = [par1 squeeze(pars(icond,5,:,ipar1))'];
        par2 = [par2 squeeze(pars(icond,5,:,ipar2))'];
    else
        for itime = 1:4
            par1 = [par1 squeeze(pars(icond,itime,:,ipar1))'];
            par2 = [par2 squeeze(pars(icond,itime,:,ipar2))'];
        end
    end
end
figure
istart = 1;
for icond = 1:3
    if ismember(icond,1:2)
        scatter(par1(istart:istart+112-1),par2(istart:istart+112-1),'MarkerFaceColor',graded_rgb(icond,4),'MarkerEdgeColor',graded_rgb(icond,4))
        istart = istart+112;
    else
        scatter(par1(istart:end),par2(istart:end),'MarkerFaceColor',graded_rgb(icond,4),'MarkerEdgeColor',graded_rgb(icond,4))
    end
    hold on
end
%plot(0:1,polyval(polyfit(par1,par2,1),0:1));

% remove lowest values of kini
idx = par1>.1;
plot(0:1,polyval(polyfit(par1(idx),par2(idx),1),0:1));

title(titletxt);
xlabel(partxt{ipar1});
ylabel(partxt{ipar2});


%% plot parameters for a given condition and time

icond = 2;
itime = 4;
figure(2)
clf
if icond == 3
    itime = 5;
end
for ip = 1:4
    subplot(1,4,ip)
    scatter(ones(1,nsubj)+normrnd(0,.05,[1 nsubj]),params(icond,itime,:,ip),'MarkerFaceColor',[.1 .1 .1],'MarkerFaceAlpha',.1,'MarkerEdgeColor','none');
    hold on
    errorbar(1,mean(params(icond,itime,:,ip)),std(params(icond,itime,:,ip),1),'o');
    xlim([.5 1.5]);
end

%% test simulation

addpath('./sim_noisyKF_paramfit')
load('subj_resp_rew_all')

icond = 2;
itime = 1;
blockrange = 4*(itime-1)+1:4*(itime-1)+4;

cfg = struct;
cfg.nb = 4;
cfg.nt = 16;
cfg.ms = .55;   cfg.vs = .07413^2; 
cfg.sbias_cor = false;  
cfg.sbias_ini = true;
cfg.cscheme = 'ths';  cfg.lscheme = 'sym';  cfg.nscheme = 'upd';
cfg.ns      = 50; % 100 simulations take around 35 seconds per subject 
cfg.sameexpe = true;

all_means = nan(28,16,4,3,2);
for icond = 1:3
    for itime = 1:4
        blockrange = 4*(itime-1)+1:4*(itime-1)+4;

        for isubj = 1:nsubj
            if icond == 3
                pars = params(icond,5,isubj,:);
            else
                pars = params(icond,itime,isubj,:);
            end

            cfg.ksi     = 0;
            cfg.epsi    = 0;
            cfg.kini    = pars(1).*ones(cfg.ns,1);
            cfg.kinf    = pars(2).*ones(cfg.ns,1);
            cfg.zeta    = pars(3).*ones(cfg.ns,1);
            cfg.theta   = pars(4).*ones(cfg.ns,1);

            cfg.compexpe    = subj_resp_rew_all(subjlist(isubj)).rew_expe(blockrange,:,icond)/100;
            cfg.firstresp   = subj_resp_rew_all(subjlist(isubj)).resp(blockrange,1,icond);

            fprintf('Simulating subject %d \n',isubj)
            out(isubj) = sim_noisyKF_fn(cfg);
        end

        out_means = nan(nsubj,cfg.nt);
        subj_means = out_means;
        for isubj = 1:nsubj
            resp = out(isubj).resp;
            resp(resp ~= 1) = 0;
            out_means(isubj,:) = mean(mean(resp,3),1);

            subj_resp = subj_resp_rew_all(subjlist(isubj)).resp(blockrange,:,icond);
            subj_resp(subj_resp ~= 1) = 0; 
            subj_means(isubj,:) = mean(subj_resp,1);
        end
        all_means(:,:,itime,icond,1) = out_means;
        all_means(:,:,itime,icond,2) = subj_means;
    end
end

%% visualize
figure
hold on
for icond = 1:3
    subplot(1,3,icond)
    hold on
    for itime = 1:4
        shadedErrorBar(1:cfg.nt,mean(all_means(:,:,itime,icond,1),1),std(all_means(:,:,itime,icond,1),1,1)/sqrt(nsubj),...
            'lineprops',{'--','Color',graded_rgb(icond,itime),'LineWidth',1.5},'patchSaturation',.1)
        errorbar(1:16,mean(all_means(:,:,itime,icond,2),1),std(all_means(:,:,itime,icond,2),1,1)/sqrt(nsubj),...
                    'CapSize',0,'Color',graded_rgb(icond,itime),'LineWidth',1.5)
    end
    ylim([.4 1])
    
end
        
        
        
        
%% sadf
shadedErrorBar(1:cfg.nt,mean(out_means,1),std(out_means,1,1)/sqrt(nsubj),...
            'lineprops',{'--','Color',graded_rgb(icond,itime),'LineWidth',1.5},'patchSaturation',.1)
hold on
errorbar(1:16,mean(subj_means,1),std(subj_means,1,1)/sqrt(nsubj),...
            'CapSize',0,'Color',graded_rgb(icond,itime),'LineWidth',1.5)
ylim([.4 1])
%%
function rgb = graded_rgb(ic,iq)
           
    red = [1.0 .92 .92; .98 .80 .80; .97 .64 .64; .96 .49 .49];
    gre = [.94 .99 .94; .85 .95 .83; .74 .91 .70; .63 .87 .58];
    blu = [.93 .94 .98; .78 .85 .94; .61 .74 .89; .44 .63 .84];
           
    rgb = cat(3,red,gre);
    rgb = cat(3,rgb,blu);
    
    rgb = rgb(iq,:,ic);
end

function rgb = param_rgb(ip)
    rgb = [ 40 36 36;...
            66 61 61;...
            52 77 91;...
            69 50 67]...
            /100;
    rgb = rgb(ip,:);
end