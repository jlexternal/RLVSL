%pupil_flexible_epoch_testing
clear all;
addpath('./Toolboxes/NoiseTools');

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

% Script parameters
% Choose locking event(1/STIM 2/RESP 3/FBCK 4/END, trial number)
ievent = 3;

%% Organize data
epoch_struct = {};

% Define window around locked point (500 samples/sec)
ep_pre  = 250;
ep_post = 2000; 
n_samp = ep_pre+ep_post+1;

ctr_excl = 0;
ctr_subj = 0;
subj_init = true;
subj_ctr = 0;
for isubj = subjlist
    subj_ctr = subj_ctr+1;
    load(sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)); % loads structure 'expe' to workspace
    ctr_subj = ctr_subj+1;
    
    % Set constants and allocate memory
    if subj_init
        nb = length(expe);
        nt = expe(1).cfg.ntrls;
        excluded_trials  = false(nt,(nb-3)/3,numel(subjlist)); % logical matrix of excluded trials
        fbs_all = nan(nt,(nb-3),numel(subjlist)); % matrix of feedback values
        sjs_all = nan(size(fbs_all));    % matrix of subject numbers
        rsp_all = nan(size(fbs_all));    % subject responses
        qts_all = nan(size(fbs_all));    % matrix of quarters
        cds_all = nan(size(fbs_all));    % matrix of conditions
        mgen = expe(1).cfg.mgen; % generative mean
        sgen = expe(1).cfg.sgen; % generative std
        imsg = nan(4*nt,nb-3,numel(subjlist)); % store event indices
        for ie = 1:4 % ie corresponds to event type
            idx_event(ie,:) = ie+4*(0:nt-1); % (index,trial number)
        end
        td_from_onset_post = nan(nb-3,nt-1,nsubj,4); % (block,trial,subj,event)
        subj_init = false;
    end
    sjs_all(:,:,subj_ctr) = subj_ctr;
    
    for ib = 4:nb
        cond = expe(ib).type;
        switch cond 
            case 'rep'
                icond = 1;
            case 'alt'
                icond = 2;
            case 'rnd'
                icond = 3;
        end
        % Fill in the other matrices
        fbs_all(:,ib-3,subj_ctr) = expe(ib).blck_trn;
        rsp(:,ib-3,subj_ctr) = expe(ib).resp;
        qts_all(:,ib-3,subj_ctr) = floor(4*(ib-3.1)/(nb-3))+1;
        cds_all(:,ib-3,subj_ctr) = icond;
        
        pupilfile = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b%02d*preproc.mat',isubj,isubj,ib));
        load(sprintf('./Data/S%02d/%s',isubj,pupilfile.name)); % loads structure 'data_eye' to workspace
        
        % Find the indices of events 
        % idx_event (1/STIM 2/RESP 3/FBCK 4/END, trial number)
        [~,imsg(:,ib-3,subj_ctr)] = pupil_event_indexer_rlvsl(data_eye);
        tsmp = data_eye.tsmp;
        psmp = data_eye.psmp;
        tmsg = data_eye.tmsg;
        
      % Detrending
        % Normalize the time sample points
        t_init = tsmp(1);
        tsmp_normd = (tsmp-t_init)/1000;

        % Identify NaN in data
        ind_nan = isnan(tsmp_normd);
        ind_nan = ind_nan | isnan(psmp);

        % Detrend data with Robust Detrending (of Alain de Cheveigné)
        polyorder = nt-1;
        pupil_detrend_rbst = nan(size(psmp));
        [fit_psmp_dtd_rbst,~,~] = nt_detrend(psmp(~ind_nan),polyorder);
        iptr  = 1;
        for is = 1:length(ind_nan)
            if ind_nan(is) == 1
            else
                pupil_detrend_rbst(is) = fit_psmp_dtd_rbst(iptr);
                iptr = iptr + 1;
            end
        end
        
        for it = 1:nt
            % Choose the window
            imsg_pre  = imsg(idx_event(ievent,it),ib-3,subj_ctr)-ep_pre;
            imsg_post = imsg(idx_event(ievent,it),ib-3,subj_ctr)+ep_post;
            epoch_struct{ib-3,it,subj_ctr} = pupil_detrend_rbst(imsg_pre:imsg_post);
            
            % Keep track of temporal differences to future events
            if it < nt
                for ie = 1:4
                    td_from_onset_post(ib-3,it,subj_ctr,ie) = imsg(idx_event(ievent,it)+ie,ib-3,subj_ctr) - imsg(idx_event(ievent,it),ib-3,subj_ctr);
                end
            end
        end
    end
    
end

% Get average times of post locked-event events
td_onset_post_means = nan(nsubj,4);
for ie = 1:4
    td_onset_post_means(:,ie) = mean(td_from_onset_post(:,:,:,ie),[1 2])/500;
end

%% Analyse: Regression beta of feedback on pupil area on BIASED and UNBIASED blocks across ALL CONDITIONS

% Load epsilon-greedy percentages
load('out_fit_epsi.mat'); % loads matrix 'epsi'
epsi(:,:,:,1) = [];
epsi(epsi<=.5) = 0;
epsi(epsi~=0) = 1;

iszscored = true; % set to true to z-score the pupil areas for any subject
baselined = false;  % baseline each epoch to before the fb onset

bvals = zeros(nsubj,n_samp,2); %(isubj,epoch,ibias)
epoch_means_subj = zeros(nsubj,n_samp);
epochs_test = [];
for isubj = 1:nsubj
    
    epochs = nan(nt*(nb-3),n_samp);
    
    % Isolate epochs for a single subject
    epoch_ctr = 1;
    nan_epoch = [];
    for ib = 1:nb-3
        for it = 1:nt
            epochs(epoch_ctr,:) = epoch_struct{ib,it,isubj};
            if sum(isnan(epochs(epoch_ctr,:))) > 0
                nan_epoch = [nan_epoch epoch_ctr];
            end
            epoch_ctr = epoch_ctr + 1;
            fbs_all(it,ib,isubj) = convert_fb_raw2seen(expe(ib+3).blck(it),expe(ib+3).resp(it),mgen,sgen)/100-.5;
        end
    end
    % Isolate experimental data
    fbs = fbs_all(:,:,isubj);
    sjs = sjs_all(:,:,isubj);
    rsp = rsp_all(:,:,isubj);
    qts = qts_all(:,:,isubj);
    cds = cds_all(:,:,isubj);
    % Vectorise
    fbs = fbs(:);
    sjs = sjs(:);
    rsp = rsp(:);
    qts = qts(:);
    cds = cds(:);
    % Remove data corresponding to NaN pupil
    epochs(nan_epoch,:) = [];
    epochs_test = cat(1,epochs_test,epochs);
    fbs(nan_epoch,:) = [];
    sjs(nan_epoch,:) = [];
    rsp(nan_epoch,:) = [];
    qts(nan_epoch,:) = [];
    cds(nan_epoch,:) = [];
    
    % Identify quarters that are biased/unbiased
    idx_bias = ones(size(qts));
    for icond = 1:3
        for iq = 1:4
            if epsi(isubj,icond,iq) == 1 % 0:unbiased, 1:biased
                idx_bias(cds==icond & qts==iq) = 2; % 1 if unbiased, 2 if biased
            end
        end
    end
    
    % Z-score within subject
    if iszscored
        epochs = zscore(epochs,[],'all'); % z-score all epochs from a subject
        zs_str = '(z-scored pupil area)';
    end
    % Baseline within subject
    if baselined
        epochs = bsxfun(@minus,epochs,mean(epochs(:,1:ep_pre),2));
    end
    
    % Regress
    for ibias = 1:2 % 1:unbiased, 2:biased
        n_ep = size(epochs(idx_bias==ibias),1);
        for isamp = 1:n_samp
            bval = regress(epochs(idx_bias==ibias,isamp),[ones(n_ep,1) fbs(idx_bias==ibias)]);
            % Note: Rank deficient warnings come from subjects who do not have biased blocks
            bvals(isubj,isamp,ibias) = bval(2);
        end
        epoch_means_subj(isubj,:,ibias) = mean(epochs(idx_bias==ibias,:),1);
    end
    
    
end
bvals(bvals == 0)=nan;
nonbiased_subjs = [];
for i = 1:4
    nonbiased_subjs = cat(2,nonbiased_subjs,epsi(:,:,i));
end
nonbiased_subjs = numel(find(sum(nonbiased_subjs,2) == 0));

%% Plot: Regression beta of feedback on pupil area on BIASED and UNBIASED blocks across ALL CONDITIONS
for i = 1:2
    if i == 2
        nsubj_temp = nsubj-nonbiased_subjs;
    else
        nsubj_temp = nsubj;
    end
    figure(1);
    shadedErrorBar(([1:n_samp]-ep_pre)/500,mean(epoch_means_subj(:,:,i),1,'omitnan'),std(epoch_means_subj(:,:,i),0,1,'omitnan')/sqrt(nsubj_temp),...
        'lineprops',{'LineWidth',2},'patchSaturation',0.075);
    hold on;
    figure(2);
    shadedErrorBar(([1:n_samp]-ep_pre)/500,mean(bvals(:,:,i),1,'omitnan'),std(bvals(:,:,i),0,1,'omitnan')/sqrt(nsubj_temp),...
        'lineprops',{'LineWidth',2},'patchSaturation',0.075);
    hold on;
end

for i = 1:2
    figure(i);
    xline(0);
    yline(0);
    legtxt = {'unbiased','biased'};
    
    text(0,0,rel_event(ievent,1,true));
    for ie = 1:4
        xline(mean(td_onset_post_means(:,ie)),':');
        text(mean(td_onset_post_means(:,ie)),0,rel_event(ievent,ie,false));
    end
    
    legend([legtxt],'Location','southwest');
    if i == 1
        title('Z-scored pupil area');
        ylabel('z-score');
    else
        title('Regression beta');
        ylabel('Beta');
    end
end


%% time difference between events
a = [];
for ib = 1:48
    for i = 2:64
        
        a(i,1) = imsg(i,ib,1)-imsg(i-1,ib);

    end
end

%% Local functions
function rgb = graded_rgb(ic,iq,nq)
    xc = linspace(.8,.2,nq);

    rgb =  [1,xc(iq),xc(iq); ...
               xc(iq),1,xc(iq); ...
               xc(iq),xc(iq),1];

    rgb = rgb(ic,:);
end

function out = rel_event(onset_event,n_post,isonsetout)
    event_str = {'STIM','RESP','FB','END'};
    if isonsetout
        out = event_str(onset_event);
    else
        if onset_event+n_post <= 4
            out = event_str(onset_event+n_post);
        else
            out = event_str(mod(onset_event+n_post,4));
        end
    end
end
