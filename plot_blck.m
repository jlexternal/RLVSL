function plot_blck(expe,iblck)
%  PLOT_BLCK  Plot block of interest from experiment structure
%
%  Usage: PLOT_BLCK(expe,iblck)
%
%  where expe is an existing experiment structure
%        iblck is the index of the block of interest
%
%  The function plots the bandit-wise mean/sampled values across trials, along
%  with the between-bandit mean values and value differences across trials.
%
%  Valentin Wyart <valentin.wyart@ens.fr> - Jan. 2016

if nargin < 2
    error('Missing input argument(s)!');
end

% extract block of interest
blck = expe(iblck);

fprintf('block parameters:\n');
fprintf('  * tau_samp = %.1f\n',blck.cfg.tau_samp);
fprintf('  * anticorr = %d\n',blck.cfg.anticorr);
fprintf('  * feedback = %d\n',blck.cfg.feedback);
fprintf('\n');

pbar = 3; % plot box aspect ratio

% show bandit values
figure;
rgb = colormap('lines');
hold on
xlim([0,97]);
ylim([0,100]);
plot(blck.vm(1,:),'-','Color',rgb(1,:)*0.5+0.5);
plot(blck.vs(1,:),'o','MarkerFaceColor',rgb(1,:),'MarkerSize',4,'Color',rgb(1,:));
plot(blck.vm(2,:),'-','Color',rgb(2,:)*0.5+0.5);
plot(blck.vs(2,:),'o','MarkerFaceColor',rgb(2,:),'MarkerSize',4,'Color',rgb(2,:));
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',14);
set(gca,'XTick',10:10:90);
set(gca,'YTick',0:20:100);
xlabel('trial number');
ylabel('bandit value');

% show mean value
figure;
rgb = colormap('lines');
hold on
xlim([0,97]);
ylim([0,100]);
plot(mean(blck.vm,1),'-','Color',0.5*[1,1,1]);
plot(mean(blck.vs,1),'ko','MarkerFaceColor',[0,0,0],'MarkerSize',4);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',14);
set(gca,'XTick',10:10:90);
set(gca,'YTick',0:20:100);
xlabel('trial number');
ylabel('mean value');

% show value difference
figure;
hold on
xlim([0,97]);
ylim([-100,100]);
plot(xlim,[0,0],'k-');
plot(diff(blck.vm,[],1),'-','Color',0.5*[1,1,1]);
plot(diff(blck.vs,[],1),'ko','MarkerFaceColor',[0,0,0],'MarkerSize',4);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',14);
set(gca,'XTick',10:10:90);
set(gca,'YTick',-100:50:+100);
xlabel('trial number');
ylabel('value difference');

end