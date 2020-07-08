% run_pupil_fb_delayed_correlation.m
% 
% Usage: Analysis of pupil areas with variables non-incident with the trial


clear all;

% subjects to be included in analysis
nsubjtot    = 31;
excluded    = [1 15 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

% Condition to analyze
condtypes = {'rep','alt','rnd'};
nc = numel(condtypes);

epc_struct = {};
cfg = [];
cfg.incl_nan = true;
for icond = 1:nc
    epc_struct{icond} = pupil_get_epochs(subjlist,condtypes{icond},cfg);
end

% choose the smallest epoch window for comparison
epc_range    = min([epc_struct{1}.epoch_window; epc_struct{2}.epoch_window; epc_struct{3}.epoch_window],[],1);
epc_fb_onset = [epc_struct{1}.epoch_window(1); epc_struct{2}.epoch_window(1); epc_struct{3}.epoch_window(1)]+1;


%% Plot: Regression beta of previous feedback value as predictor of pupil area along epoch (delayed correlation)

tback = 1; % feedback of how many trials back to analyze
iszscored = true;

leg_str =[];
figure;
for icond = 1:nc
    epochs = epc_struct{icond}.epochs;
    plot_window  = epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2);
    epoch_window = epc_struct{icond}.epoch_window;
    epochs = epochs(:,plot_window);
    
    idx_subj = epc_struct{icond}.idx_subj;
    ind_nan  = epc_struct{icond}.ind_epoch_nan;
    fbs = epc_struct{icond}.fbs/100;
    bks = epc_struct{icond}.bks;
    trs = epc_struct{icond}.trs;
    nt = max(trs(1:16));
    
    % Identify pupil traces and fb that are tback from those pupil traces
    ind_trl_ep = trs > tback;       % trials for pupil
    ind_trl_fb = trs <= nt-tback;   % trials-tback for feedback
    
    bvals      = zeros(nsubj,size(epochs,2));
    bvals_orig = zeros(nsubj,size(epochs,2));
    
    for isubj = 1:nsubj
        ind_subj    = idx_subj == isubj;
        ind_epoch   = ind_subj & ind_trl_ep;
        ind_fbs     = ind_subj & ind_trl_fb;
        ind_orig    = ind_subj & ~ind_nan;
        n_eps       = size(epochs(ind_epoch,:),1);
        n_eps_orig  = size(epochs(ind_orig,:),1);
        
        if iszscored
            epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
            zs_str = '(z-scored pupil area)';
        end
    
        for isamp = 1:size(epochs,2)
            % correlation of pupil with feedback from tback trials back
            bval = regress(epochs(ind_epoch,isamp),[ones(n_eps,1) fbs(ind_fbs)]);
            bvals(isubj,isamp) = bval(2);
            % correlation of pupil with trial-linked feedback
%            bval_orig = regress(epochs(ind_orig,isamp),[ones(n_eps_orig,1) fbs(ind_orig)]);
%            bvals_orig(isubj,isamp) = bval_orig(2);
            %warning('off','all');
        end
    end
    err_bar = std(bvals,[],1)/sqrt(nsubj);
%    err_bar_orig = std(bvals_orig,[],1)/sqrt(nsubj);
    shade_str = 'SEM';
    xaxis = ((1:size(bvals,2))-epoch_window(1))*2/1000;
    shadedErrorBar(xaxis,mean(bvals),err_bar,...
                   'lineprops',{':','LineWidth',2,'Color',graded_rgb(icond,1,1)},'patchSaturation',0.075);
%    shadedErrorBar(xaxis,mean(bvals_orig),err_bar_orig,...
%                   'lineprops',{'LineWidth',2,'Color',graded_rgb(icond,1,1)},'patchSaturation',0.075);
    leg_str = [leg_str,string(condtypes{icond})];
    hold on;
end
legend(leg_str);
xline(0,'LineWidth',2,'HandleVisibility','off');
xlabel('Time around fb onset (s)');
yline(0,'HandleVisibility','off');
ylabel('Regression Beta');
xlim([min(xaxis) max(xaxis)]);
ylim([-1.5 1.5])
title_str = sprintf('Reg. coef. beta %s around fb onset to fb at trial-%d \nacross all subjs & conditions\nShaded bars: %s',...
                    zs_str,tback,shade_str);
title(title_str);
%% Local functions
function rgb = graded_rgb(ic,iq,nq)
    xc = linspace(.8,.2,nq);

    rgb =  [1,xc(iq),xc(iq); ...
               xc(iq),1,xc(iq); ...
               xc(iq),xc(iq),1];

    rgb = rgb(ic,:);
end
