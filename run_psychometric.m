
%% Load data

% clear workspace
clear all java
close all hidden
clc

% define list of subjects
% excluded:
%   * S01 because paradigm tweaked after
%   * S23 and S28 because p(correct) < 0.5 for novel bandits
subjlist = setdiff(01:31,[01,23,28]);
nsubj = numel(subjlist);

% define list of conditions (do not change!)
condlist = {'rep','alt','rnd'};

% create data structure
dat      = [];
dat.subj = []; % subject number (1 through n)
dat.cond = []; % condition (1:rep 2:alt 3:rnd)
dat.qrt  = []; % condition-wise quartile (1 through 4)
dat.blk  = []; % condition-wise block number (1 through 16)
dat.trl  = []; % block-wise trial number (1 through 16)
dat.resp = []; % response (1:high-reward 2:low-reward)
dat.rtim = []; % response time (in sec)
dat.r_hi = []; % reward associated with high-reward option

for isubj = 1:nsubj
    
    % load experiment data file
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat', ...
        subjlist(isubj),subjlist(isubj));
    if ~exist(filename,'file')
        error('missing experiment data file!');
    end
    load(filename,'expe');
    
    % parse experiment data
    for icond = 1:3
        iblk = find(cellfun(@(s)strcmp(s,condlist{icond}),{expe.type}));
        nblk = numel(iblk);
        if nblk ~= 16
            error('weird number of blocks found!');
        end
        for jblk = 1:nblk
            dat.subj = cat(1,dat.subj,isubj*ones(16,1));
            dat.cond = cat(1,dat.cond,icond*ones(16,1));
            dat.qrt  = cat(1,dat.qrt,ceil(jblk/4)*ones(16,1));
            dat.blk  = cat(1,dat.blk,jblk*ones(16,1));
            dat.trl  = cat(1,dat.trl,(1:16)');
            dat.resp = cat(1,dat.resp,expe(iblk(jblk)).resp(:));
            dat.rtim = cat(1,dat.rtim,expe(iblk(jblk)).rt(:));
            dat.r_hi = cat(1,dat.r_hi,expe(iblk(jblk)).blck_trn(:));
        end
    end
    
end

% clean workspace from temporary variables
clearvars -except subjlist nsubj condlist dat

%% Fit exponential learning curve to single-subject data

close all hidden
clc

use_prf = true; % use prior functions on parameter values?

% subject/best-fitting learning curves
pcor_sub = nan(nsubj,16,3,4); % subject learning curve
pcor_fit = nan(nsubj,16,3,4); % best-fitting learning curve

% best-fitting parameter values
pini_fit = nan(nsubj,3,4); % initial performance
pinf_fit = nan(nsubj,3,4); % asymptotic performance
lrat_fit = nan(nsubj,3,4); % learning rate

hbar = waitbar(0,'');
for isubj = 1:nsubj
    waitbar(isubj/nsubj,hbar,sprintf('processing S%02d/%02d',isubj,nsubj));
    for icond = 1:3
        for itime = 1:4
        
            % filter data
            ifilt = dat.subj == isubj & dat.cond == icond & dat.qrt == itime;

            % fit exponential learning curve
            cfg         = [];
            cfg.trl     = dat.trl(ifilt);
            cfg.resp    = dat.resp(ifilt);
            cfg.use_prf = use_prf;
            out = fit_explrn(cfg);
            
            % compute subject learning curve
            pcor_sub(isubj,:,icond,itime) = grpstats(cfg.resp == 1,cfg.trl,@mean);
            
            for itrl = 2:16
                jfilt = find(ifilt & dat.trl == itrl);
                prep_sub(isubj,itrl,icond,itime) = mean(dat.resp(jfilt) == dat.resp(jfilt-1));
            end
            
            % get best-fitting learning curve
            pcor_fit(isubj,:,icond,itime) = out.plrn;
            
            % get best-fitting parameter values
            pini_fit(isubj,icond,itime) = out.pini;
            pinf_fit(isubj,icond,itime) = out.pinf;
            lrat_fit(isubj,icond,itime) = out.lrat;
            
        end
    end
end
close(hbar);

%% Plot subject and best-fitting learning curves

close all hidden
clc

icond = 3; % condition of interest

% compute statistics for subject learning curves
psub_avg = squeeze(mean(pcor_sub(:,:,icond,:),1))';
psub_err = squeeze(std(pcor_sub(:,:,icond,:),[],1))'/sqrt(nsubj);

% compute statistics for best-fitting learning curves
pfit_avg = squeeze(mean(pcor_fit(:,:,icond,:),1))';
pfit_err = squeeze(std(pcor_fit(:,:,icond,:),[],1))'/sqrt(nsubj);

itrl = 1:16; % x-axis

% define colors for condition of interest
rgb = zeros(1,3); rgb(icond) = 1;
rgb = bsxfun(@plus,bsxfun(@times,rgb,(1:4)'/4),(3:-1:0)'/4);
pbar = 4/3; % plot box aspect ratio
figure;
hold on
xlim([0.5,16.5]);
ylim([0.375,1]);
% plot shaded error bars for best-fitting learning curves
for i = 1:4
    patch([itrl,fliplr(itrl)], ...
        [pfit_avg(i,:)+pfit_err(i,:),fliplr(pfit_avg(i,:)-pfit_err(i,:))], ...
        0.5*(rgb(i,:)+1),'EdgeColor','none');
end
plot(xlim,[1,1]*0.5,'-','Color',[1,1,1]*0.8);
% plot best-fitting learning curves (lines)
for i = 1:4
    plot(itrl,pfit_avg(i,:),'-','Color',rgb(i,:),'LineWidth',1);
end
% plot error bars for subject learning curves
for i = 1:4
    for j = 1:16
        plot(j*[1,1],psub_avg(i,j)+psub_err(i,j)*[+1,-1],'-', ...
            'Color',rgb(end,:));
    end
end
% plot subject learning curves (dots)
for i = 1:4
    plot(itrl,psub_avg(i,:),'o','MarkerSize',4, ...
        'Color',rgb(end,:),'MarkerFaceColor',rgb(i,:));
end
hold off
% parameterize axes
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',2:2:16);
set(gca,'YTick',0:0.1:1);
xlabel('trial position in block','FontSize',8);
ylabel('fraction correct','FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end

%% Plot initial and asymptotic accuracies across block positions

close all hidden
clc

icond = 3; % condition of interest

% perform repeated-measures ANOVA
tbl = simple_mixed_anova( ...
    cat(2,pini_fit(:,icond,:),pinf_fit(:,icond,:)),[], ...
    {'ini_inf','block'})

% define colors for condition of interest
rgb = zeros(1,3); rgb(icond) = 1;
rgb = bsxfun(@plus,bsxfun(@times,rgb,(1:4)'/4),(3:-1:0)'/4);
pbar = 1; % plot box aspect ratio
wid = 0.4; % violin width
figure;
hold on
xlim([0.4,4.6]);
ylim([0,1]);
plot(xlim,[1,1]*0.5,'-','Color',[1,1,1]*0.8);
for itime = 1:4
    pos = itime; % position on x-axis
    % compute violin for initial accuracy
    x = squeeze(pini_fit(:,icond,itime));
    xk = linspace(min(x),max(x),100);
    pk = ksdensity(x,xk); pk = pk/max(pk);
    sx = interp1(xk,pk,x);
    jx = linspace(-1,+1,numel(x));
    % compute violin for asymptotic accuracy
    y = squeeze(pinf_fit(:,icond,itime));
    yk = linspace(min(y),max(y),100);
    qk = ksdensity(y,yk); qk = qk/max(qk);
    sy = interp1(yk,qk,y);
    jy = linspace(-1,+1,numel(y));
    % plot violins
    for i = 1:numel(x)
        plot(pos+jx(i)*sx(i)*wid,x(i),'wo','MarkerFaceColor',0.5*(rgb(itime,:)+1), ...
            'MarkerSize',4,'LineWidth',0.5);
        plot(pos+jy(i)*sy(i)*wid,y(i),'wo','MarkerFaceColor',0.25*rgb(itime,:)+0.75, ...
            'MarkerSize',4,'LineWidth',0.5);
    end
end
% plot means
plot(1:4,squeeze(mean(pini_fit(:,icond,:),1)),'-', ...
    'LineWidth',0.75,'Color',rgb(end,:));
plot(1:4,squeeze(mean(pinf_fit(:,icond,:),1)),'-', ...
    'LineWidth',0.75,'Color',0.5*(rgb(end,:)+1));
for itime = 1:4
    pos = itime; % position on x-axis
    x = squeeze(pini_fit(:,icond,itime));
    % plot error bar
    plot(pos*[1,1],mean(x)+std(x)/sqrt(numel(x))*[-1,+1],'-', ...
        'Color',rgb(end,:));
    % plot mean
    plot(pos,mean(x),'o','MarkerSize',5,'LineWidth',0.5, ...
        'Color',rgb(end,:),'MarkerFaceColor',rgb(itime,:));
end
for itime = 1:4
    pos = itime; % position on x-axis
    x = squeeze(pinf_fit(:,icond,itime));
    % plot error bar
    plot(pos*[1,1],mean(x)+std(x)/sqrt(numel(x))*[-1,+1],'-', ...
        'Color',0.5*(rgb(end,:)+1));
    % plot mean
    plot(pos,mean(x),'o','MarkerSize',5,'LineWidth',0.5, ...
        'Color',0.5*(rgb(end,:)+1),'MarkerFaceColor',0.5*(rgb(itime,:)+1));
end
hold off
% parameterize axes
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',1:4,'XTickLabel',{'Q1','Q2','Q3','Q4'});
set(gca,'YTick',0:0.2:1);
xlabel('block position','FontSize',8);
ylabel('accuracy','FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
