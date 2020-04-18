clear all;

% subjects to be included in analysis
nsubjtot    = 31;
excluded    = [1 15 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);

% Condition to analyze
condtypes = {'rep','alt','rnd'};

epc_struct = {};

for icond = 1:numel(condtypes)
    epc_struct{icond} = pupil_get_epochs(subjlist,condtypes{icond});
end

% choose the smallest epoch window for comparison
epc_range    = min([epc_struct{1}.epoch_window; epc_struct{2}.epoch_window; epc_struct{3}.epoch_window],[],1);
epc_fb_onset = [epc_struct{1}.epoch_window(1); epc_struct{2}.epoch_window(1); epc_struct{3}.epoch_window(1)]+1;


%% Plot: Regression beta of feedback value as predictor of pupil area along the epoch
%           Within condition comparisons

figure(1);
for icond = 1:numel(condtypes)
    epochs = epc_struct{icond}.epochs;
    plot_window  = epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2);
    epoch_window = epc_struct{icond}.epoch_window;
    epochs = epochs(:,plot_window);
    epochs = zscore(epochs,[],2);
    
    fbs = epc_struct{icond}.fbs;
    
    %
    % to do: get per-subject means instead of on the entire dataset
    %
    bvals = zeros(1,size(epochs,2));
    bints = zeros(2,size(epochs,2));
    for isamp = 1:size(epochs,2)
        [bvals(isamp),bints(:,isamp)] = regress(epochs(:,isamp),fbs);
    end
    bints = flip(bints,1); % reverse order of lo/up->up/lo conf bounds for plotting
    bint_bar = abs(bints-repmat(bvals,[2,1]));
    xaxis = ((1:numel(bvals))-epoch_window(1))*2/1000;
    shadedErrorBar(xaxis,bvals,bint_bar,...
                    'lineprops',{'LineWidth',2,'Color',graded_rgb(icond,1,1)},'patchSaturation',0.075);
    %plot(xaxis,bvals,'LineWidth',2,'Color',graded_rgb(icond,1,1));
    hold on;
end
legend(condtypes);
xline(0,'LineWidth',2,'HandleVisibility','off');
xlabel('Time around fb onset (s)');
yline(0,'HandleVisibility','off');
ylabel('Regression Beta');
xlim([min(xaxis) max(xaxis)]);
titlestr = sprintf('Evolution of reg. coef. beta within epoch around fb onset across all subjs & conditions\nShaded bars: 95%% C.I.');
title(titlestr);
%% Plot: Pupil dilation grouped in to pos and neg feedback

figure(2);
for icond = 1:numel(condtypes)
    epochs = epc_struct{icond}.epochs;
    plot_window = (epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2));
    epoch_window = epc_struct{icond}.epoch_window;
    epochs = epochs(:,plot_window);
    idx_subj_epoch = epc_struct{icond}.idx_subj;
    fbs = epc_struct{icond}.fbs;
    
    epoch_mean_pos = [];
    epoch_mean_neg = [];

    for isubj = 1:numel(subjlist)
        idx_subj = idx_subj_epoch == isubj;
        epoch_mean_pos(isubj,:) = mean(zscore(epochs(idx_subj & fbs>=0,:),[],2),1);
        epoch_mean_neg(isubj,:) = mean(zscore(epochs(idx_subj & fbs<0,:),[],2),1);
    end

    xaxis = ((1:size(epochs,2))-epoch_window(1))*2/1000;
    subplot(3,1,icond);
    yline(.4,':','HandleVisibility','off');
    yline(-.4,':','HandleVisibility','off');
    shadedErrorBar(xaxis,mean(epoch_mean_pos,1),std(epoch_mean_pos)/sqrt(numel(subjlist)),...
                    'lineprops',{'Color',[.8 .2 .2],'LineWidth',2},'patchSaturation',0.075);
    hold on;
    shadedErrorBar(xaxis,mean(epoch_mean_neg,1),std(epoch_mean_neg)/sqrt(numel(subjlist)),...
                    'lineprops',{'Color',[.2 .2 .8],'LineWidth',2},'patchSaturation',0.075);
    legend('pos. fb','neg. fb');
    xline(0,'LineWidth',2,'HandleVisibility','off');
    yline(0,'HandleVisibility','off');
    xlabel('Time around fb onset (s)');
    ylabel('Z-scored pupil area (a.u.)');
    xlim([min(xaxis) max(xaxis)]);
    ylim([-.6 .6]);
    title(sprintf(['Evolution of pupil area around onset of feedback\n(%s condition); %d subjects,\n' ...
                    'shaded error: SEM'],condtypes{icond},numel(subjlist)));
end

%% Organize: Pupil dilation across quarters

epoch_mean_qt = {}; % {quarter}(subj,trend,condition)
for icond = 1:numel(condtypes)
    epochs = epc_struct{icond}.epochs;
    plot_window = (epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2));
    epoch_window = epc_struct{icond}.epoch_window;
    epochs = epochs(:,plot_window);
    idx_subj_epoch = epc_struct{icond}.idx_subj;
    fbs = epc_struct{icond}.fbs;
    qts = epc_struct{icond}.qts;
    
    for iq = 1:4
        for isubj = 1:numel(subjlist)
            idx_subj = idx_subj_epoch == isubj;
            epoch_mean_qt{iq}(isubj,:,icond) = mean(zscore(epochs(idx_subj & qts==iq,:),[],2),1);
        end
    end
end

% Plot: grouped by condition
xaxis = ((1:size(epochs,2))-epoch_window(1))*2/1000;
figure(3);
hold on;
for icond = 1:numel(condtypes)
    
    subplot(3,1,icond);
    xline(0,'LineWidth',2,'HandleVisibility','off');
    yline(0,'HandleVisibility','off');
    yline(.4,':','HandleVisibility','off');
    yline(-.4,':','HandleVisibility','off');
    for iq = 1:4
        shadedErrorBar(xaxis,mean(epoch_mean_qt{iq}(:,:,icond),1),std(epoch_mean_qt{iq}(:,:,icond))/sqrt(numel(subjlist)),...
                    'lineprops',{'Color',graded_rgb(icond,iq,4),'LineWidth',2},'patchSaturation',0.075);
    end
    legend('Q1','Q2','Q3','Q4');
    xlabel('Time around fb onset (s)');
    ylabel('Z-scored pupil area (a.u.)');
    xlim([min(xaxis) max(xaxis)]);
    ylim([-.6 .6]);
    title(sprintf(['Evolution of pupil area around onset of feedback split across quarters\n(%s condition); %d subjects,\n' ...
                    'shaded error: SEM'],condtypes{icond},numel(subjlist)));
end

    
            
%% Local functions
function rgb = graded_rgb(ic,iq,nq)
    xc = linspace(.8,.2,nq);

    rgb =  [1,xc(iq),xc(iq); ...
               xc(iq),1,xc(iq); ...
               xc(iq),xc(iq),1];

    rgb = rgb(ic,:);
end
