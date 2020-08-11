% run_pupil_rt_analysis
%
% Usage: Analysis of pupillometric measures with reaction times

clear all;

% subjects to be included in analysis
nsubjtot    = 31;
excluded    = [1 15 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

% Condition to analyze
condtypes = ["rep","alt","rnd"];
nc = numel(condtypes);

%%%% Organize data for RTs

filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',2,2);
load(filename,'expe');
nt = expe(1).cfg.ntrls;
nb = expe(1).cfg.nbout;
ntrain = expe(1).cfg.ntrain;
nb_c = nb/nc;

blcks       = nan(nb_c,nt,3,nsubj);
resps       = nan(nb_c,nt,3,nsubj);
rts         = nan(nb_c,nt,3,nsubj);

mu_new   = 55;  % mean of higher distribution
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

a = sig_opti/expe(1).cfg.sgen;      % slope of linear transformation aX+b
b = mu_new - a*expe(1).cfg.mgen;    % intercept of linear transf. aX+b

for isubj = subjlist
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj);
    fprintf('Loading subject %d...\n',isubj);
    load(filename,'expe');
    ib_c = ones(3,1);
    for ib = 1+ntrain:nb+ntrain
        ctype = expe(ib).type;
        switch ctype
            case 'rnd' % random across successive blocks
                icond = 3;
            case 'alt' % always alternating
                icond = 2;
            case 'rep' % always the same
                icond = 1;
        end

        resp_mult = -(expe(ib).resp-1.5)*2;
        blcks(ib_c(icond),:,icond,isubj) = round(resp_mult.*expe(ib).blck*a+b);
        
        rts(ib_c(icond),:,icond,isubj) = expe(ib).rt;
        ib_c(icond) = ib_c(icond)+1;
    end
end
blcks = (blcks-50)/100;


%%%% Get pupil epochs

epc_struct = {};
% Check pupil_get_epochs.m for optional config.
usecfg = true;
if usecfg
    cfg = struct;
    %cfg.polyorder = 20; % default is nt
    cfg.r_ep_lim = 4; % set the rightward limit of epoch window
end
for icond = 1:nc
    if usecfg
        epc_struct{icond} = pupil_get_epochs(subjlist,condtypes{icond},cfg);
    else
        epc_struct{icond} = pupil_get_epochs(subjlist,condtypes{icond});
    end
end
% choose the smallest epoch window for comparison
epc_range    = min([epc_struct{1}.epoch_window; epc_struct{2}.epoch_window; epc_struct{3}.epoch_window],[],1);
epc_fb_onset = [epc_struct{1}.epoch_window(1); epc_struct{2}.epoch_window(1); epc_struct{3}.epoch_window(1)]+1;

ifig = 0;

%% 1a. Organize: Correlation between feedback magnitude and RT

coefs_rt_fb = zeros(nsubj,4,nc); 
pvals = zeros(nc,4);

for icond = 1:nc
    for iq = 1:4
        for isubj = 1:nsubj
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            rts_vec = rts(blockrange,2:nt,icond,subjlist(isubj)); % rts on trial t
            rts_vec = rts_vec(:);
            rsp_vec = blcks(blockrange,1:nt-1,icond,subjlist(isubj)); % fb on trial t-1
            rsp_vec = rsp_vec(:);
            ind_negfb = rsp_vec < 0; % consider only neg fb
            rsp_vec = rsp_vec(ind_negfb);
            bval = regress(rts_vec(ind_negfb),[ones(size(rsp_vec)) rsp_vec]);
            coefs_rt_fb(isubj,iq,icond) = bval(2);
            
        end 
        [~,pvals(icond,iq)] = ttest(coefs_rt_fb(:,iq,icond));
    end
end

%% 1b. Organize: Regress pupil area to (negative) reward
iszscored = true;
baselined = true;

for icond = 1:nc
    epochs       = epc_struct{icond}.epochs;
    plot_window  = (epc_fb_onset(icond)-epc_range(1):epc_fb_onset(icond)+epc_range(2));
    epoch_window = epc_struct{icond}.epoch_window;
    epochs       = epochs(:,plot_window);
    idx_subj     = epc_struct{icond}.idx_subj;
    qts          = epc_struct{icond}.qts;
    fbs          = epc_struct{icond}.fbs/100;
    
    ind_negfb    = fbs < 0; % index negative reward trials
    
    if icond == 1
        epoch_means = zeros(nsubj,size(epochs,2),4,nc); % (subjects,epochlength,quarters,conditions)
        bvals_qt  	= zeros(nsubj,size(epochs,2),4,nc); 
    end
    
    for isubj = 1:nsubj
        ind_subj = idx_subj == isubj;
        if iszscored
            epochs(ind_subj,:) = zscore(epochs(ind_subj,:),[],'all'); % z-score all epochs from a subject
            zs_str = '(z-scored pupil area)';
        end
        if baselined
            epochs(ind_subj,:) = bsxfun(@minus,epochs(ind_subj,:),mean(epochs(ind_subj,1:epoch_window(1)),2));
        end
        
        for iq = 1:4
            ind_subj = idx_subj == isubj;
            n_eps = size(epochs(ind_subj & ind_negfb & qts==iq,:),1);
            for isamp = 1:size(epochs,2)
                bval = regress(epochs(ind_subj & ind_negfb & qts==iq,isamp),[ones(n_eps,1) fbs(ind_subj & ind_negfb & qts==iq)]);
                bvals_qt(isubj,isamp,iq,icond) = bval(2);
            end
        end
    end
end

%% Regress pupil differences (baseline Q1) to RT coeff differences (baseline Q1)

for icond = 1:nc
    
    for isubj = 1:nsubj
        
        for iq = 2:4
            
            
        end
        
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
