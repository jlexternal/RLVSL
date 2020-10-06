% run_model_selection_KFunbiased
%
% Objective: Compare the different fitting options for the unbiased Kalman filter
%             model on the random condition and choose which set of options best fit the
%             subjects' data.
%
% Jun Seok Lee <jlexternal@gmail.com>

% loop through and store in local memory
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);
load('subj_resp_rew_all.mat'); % load all relevant subject data

% fitting options
cschemes = {'qvs','ths'};
lschemes = {'sym','ind'};
nschemes = {'rpe','upd'};
ic = 3;

params = nan(3,nsubjtot); % kini,kinf,zeta 

vec_combis = combvec(1:length(cschemes),1:length(lschemes),1:length(nschemes));
noptions = size(vec_combis,2);
schemestr = cell(1,noptions);

% organize data
ic = 3; 
out_fit_rnd = struct;
elbos   = nan(noptions,nsubj);
lls     = nan(noptions,nsubj);
for i = 1:noptions
    % define options
    out_fit_rnd(i).cscheme = cschemes{vec_combis(1,i)};
    out_fit_rnd(i).lscheme = lschemes{vec_combis(2,i)};
    out_fit_rnd(i).nscheme = nschemes{vec_combis(3,i)};
    
    % extract data from subject data files
    schemestr{i} = sprintf('%s_%s_%s',cschemes{vec_combis(1,i)},lschemes{vec_combis(2,i)},nschemes{vec_combis(3,i)});
    jsubj = 1;
    for isubj = subjlist
        filename = sprintf('./options_fit/out_fit_KFunbiased_%s_%d_%02d.mat',schemestr{i},nsubj,isubj);
        load(filename);
        % store fitted parameters
        out_fit_rnd(i).params(1,jsubj) = out_fit{ic,isubj}.kini;
        out_fit_rnd(i).params(2,jsubj) = out_fit{ic,isubj}.kinf;
        out_fit_rnd(i).params(3,jsubj) = out_fit{ic,isubj}.zeta;
        % store model evidence 
        out_fit_rnd(i).elbo(jsubj)  = out_fit{ic,isubj}.elbo;
        out_fit_rnd(i).ll(jsubj)    = out_fit{ic,isubj}.l;
        jsubj = jsubj + 1;
    end
    
    elbos(i,:)  = out_fit_rnd(i).elbo;
    lls(i,:)    = out_fit_rnd(i).ll;
end

%% Fixed-effects Bayesian model selection

% winning model for each subject based on max ELBO
[elbos_subj_descend,i_elbos_subj] = sort(elbos,1,'descend');

figure(1)
hold on;
histogram(i_elbos_subj(1,:));
title('Winning model for each subject based on ELBO');
ticklabels = sprintf('%s\\newline%s\\newline%s\n',...
                    [convertCharsToStrings(extractBetween(schemestr,1,3));...
                    convertCharsToStrings(extractBetween(schemestr,5,7));...
                    convertCharsToStrings(extractBetween(schemestr,9,11))]);
xticklabels(ticklabels);
xlabel('Unbiased KF model options')
hold off;

% winning model when pooled together
% rank the models
[sum_elbos,i_elbo] = sort(sum(elbos,2),'descend');
iwinner_pooled = i_elbo(1);
% calculate Bayes factor for winning model (pooled)
elbos_pooled = sum(elbos,2);
bayesfac_pooled = nan(noptions,1);
for iopt = 1:noptions
    if iopt == iwinner_pooled
        bayesfac_pooled(iopt) = 0;
        continue
    end
    bayesfac_pooled(iopt) = exp(elbos_pooled(iwinner_pooled)-elbos_pooled(iopt));
end
figure(2)
hold on
bar(log10(bayesfac_pooled))
xticks(1:noptions)
ticklabels = sprintf('%s\\newline%s\\newline%s\n',...
                    [convertCharsToStrings(extractBetween(schemestr,1,3));...
                    convertCharsToStrings(extractBetween(schemestr,5,7));...
                    convertCharsToStrings(extractBetween(schemestr,9,11))]);
xticklabels(ticklabels)
ylabel('Log10(Bayes Factor)')
xlabel('Unbiased KF model options')
title('Model comparison (all subjects pooled)')
hold off

% calculate Bayes factor for winning model (for each subject)
bayesfacs = nan(noptions,nsubj);
for isubj = 1:nsubj
    iwinner = i_elbos_subj(1,isubj);
    
    bayesfacs(iwinner,isubj) = 1;
    for iopt = setdiff(1:noptions,iwinner)
        bayesfacs(iopt,isubj) = exp(elbos(iwinner,isubj)-elbos(iopt,isubj));
    end
end
figure(3)
hm = heatmap(log10(bayesfacs));
colorbar
colormap([linspace(1,.4,4)' linspace(1,.4,4)' ones(4,1)]);
caxis([0 2]);
ylabel('Unbiased KF model option set');
hm.YDisplayLabels = convertCharsToStrings(schemestr);
xlabel('Subject');
title(sprintf('Model comparison (single subj) \n Log_10 of Bayes Factor'))


%% Random-effects Bayesian Model Selection
addpath('../../Toolboxes/spm12')
[alpha,pexp,pexc,pxp] = spm_BMS(elbos');

% > Winning model : choice stochasticity by Thompson sampling, 
%                   symmetric learning updates (not independent)
%                   noise scaling on the update value (not rpe)

