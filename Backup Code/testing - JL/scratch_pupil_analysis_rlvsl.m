% test_pupil_analysis_rlvsl
clear all;

% Add toolbox paths
addpath('./Toolboxes/NoiseTools/');

cfg = [];
% subjects to be included in analysis
nsubjtot    = 31;
excluded    = [1 15 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);

% Condition to analyze
condtype = 'rnd';

% Find the epoch window that fits all subjects
[epoch_window,excluded_trials,fbs] = pupil_epoch_window_rlvsl(subjlist,condtype,cfg);
fbs = fbs-50; % make the negative feedback actually negative
nt = size(fbs,1);
nb = size(fbs,2);
n_excld = sum(sum(sum(excluded_trials)));
epochs = nan(numel(fbs)-n_excld,sum(epoch_window)+1);
idx_subj_epoch = zeros(numel(fbs)-n_excld,1);

%% Organize the data
ctr_epoch = 0;
ctr_subj = 0;
for isubj = subjlist
    ctr_subj = ctr_subj+1;
    load(sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)); % loads structure 'expe' to workspace
    
    ctr_blck = 0;
    for ib = 1:length(expe)
        if strcmpi(expe(ib).type,condtype)
            ctr_blck = ctr_blck+1;
            
            % Load the pupil data
            pupilfile = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b%02d*preproc.mat',isubj,isubj,ib));
            load(sprintf('./Data/S%02d/%s',isubj,pupilfile.name)); % loads structure 'data_eye' to workspace
            tsmp = data_eye.tsmp;
            psmp = data_eye.psmp;
            tmsg = data_eye.tmsg;
            
            % Detrend the data
                % Normalize the time sample points
                t_init = tsmp(1);
                tsmp_normd = (tsmp-t_init)/1000;

                % Identify NaN in data
                idx_nan = isnan(tsmp_normd);
                idx_nan = idx_nan | isnan(psmp);
                
                % Detrend data with Robust Detrending (of Alain de Cheveigné)
                pupil_detrend_rbst = nan(size(psmp));
                [fit_psmp_dtd_rbst,~,~] = nt_detrend(psmp(~idx_nan),nt);
                
                % need to fill in gaps or ignore the trials with massive nan in the
                % series

                iptr  = 1;
                for is = 1:length(idx_nan)
                    if idx_nan(is) == 1

                    else
                        pupil_detrend_rbst(is) = fit_psmp_dtd_rbst(iptr);
                        iptr = iptr + 1;
                    end
                end
            
            % Find the indices of events 
            % idx ( : , 1/STIM 2/RESP 3/FBCK 4/END)
            [idx,imsg] = pupil_event_indexer_rlvsl(data_eye);
            idx = idx(:,3); % keep only the feedback indices
            imsg = imsg(3+4*(0:nt-1));
            
            % Extract epochs
            for it = 1:nt
                if excluded_trials(it,ctr_blck,ctr_subj)
                    continue
                end
                ctr_epoch = ctr_epoch+1;
                tstart = imsg(it) - epoch_window(1);
                tend   = imsg(it) + epoch_window(2);
                epochs(ctr_epoch,:) = pupil_detrend_rbst(tstart:tend); % log the epoch
                idx_subj_epoch(ctr_epoch) = ctr_subj;    % log the relative subj number
            end
        end
    end
end

excluded_trials = excluded_trials(:);
fbs = fbs(:);
fbs = fbs(~excluded_trials);

%% Plot: Regression beta of feedback value as predictor of pupil area along the epoch

idx_epoch_nan = false(size(epochs,1),1);

for ie = 1:size(epochs,1) % go through each epoch
    if sum(isnan(epochs(ie,:)))>0
        idx_epoch_nan(ie) = 1;
    end
end
epochs_nonan = epochs(~idx_epoch_nan,:);
fbs_nonan = fbs(~idx_epoch_nan);
bvals = zeros(1,size(epochs,2));
for isamp = 1:size(epochs,2)
    bvals(isamp) = regress(epochs_nonan(:,isamp),fbs_nonan);
end
xaxis = ((1:numel(bvals))-epoch_window(1))*2/1000;
figure(1);
plot(xaxis,bvals,'LineWidth',2);
hold on;
xline(0,'LineWidth',2,'HandleVisibility','off');
xlabel('Time around fb onset');
yline(0,'HandleVisibility','off');
ylabel('Regression Beta');
xlim([min(xaxis) max(xaxis)]);
title('Evolution of reg. coeff. beta within epoch around fb onset');
hold off;

%% Plot: Pupil dilation grouped in to pos and neg feedback
epoch_mean_pos = [];
epoch_mean_neg = [];
idx_subj_epoch_nonan = idx_subj_epoch(~idx_epoch_nan,:);

for isubj = 1:numel(subjlist)
    idx_subj = idx_subj_epoch_nonan == isubj;
    epoch_mean_pos(isubj,:) = mean(zscore(epochs_nonan(idx_subj & fbs_nonan>=0,:),[],2),1);
    epoch_mean_neg(isubj,:) = mean(zscore(epochs_nonan(idx_subj & fbs_nonan<0,:),[],2),1);
end

figure(2);
shadedErrorBar(xaxis,mean(epoch_mean_pos,1),std(epoch_mean_pos)/sqrt(numel(subjlist)),...
                'lineprops',{'Color',[.8 .2 .2],'LineWidth',2},'patchSaturation',0.075);
hold on;
shadedErrorBar(xaxis,mean(epoch_mean_neg,1),std(epoch_mean_neg)/sqrt(numel(subjlist)),...
                'lineprops',{'Color',[.2 .2 .8],'LineWidth',2},'patchSaturation',0.075);
legend('pos. fb','neg. fb');
xline(0,'LineWidth',2,'HandleVisibility','off');
yline(0,'HandleVisibility','off');
xlabel('Time around fb onset');
ylabel('Pupil area');
xlim([min(xaxis) max(xaxis)]);
title(sprintf(['Evolution of pupil area around onset of feedback (%s condition)\n %d subjects, ' ...
                '%d/%d trials analyzed due to NaN issues\n shaded error: SEM'],condtype,numel(subjlist),size(fbs_nonan,1),size(fbs,1)));



