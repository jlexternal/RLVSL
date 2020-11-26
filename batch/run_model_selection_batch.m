% run_model_selection_batch_KFbiased
%
% Objective: Compare the different fitting options for the unbiased Kalman filter
%             model on the random condition and choose which set of options best fit the
%             subjects' data.
%
% Version:   Code for fits from the fit_noisyKF_epsibias.m fitting code (pre 12 Oct 2020)
%
% Jun Seok Lee <jlexternal@gmail.com>
clc
% loop through and store in local memory
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

load('subj_resp_rew_all.mat'); % load all relevant subject data

addpath('./param_fit_PF_KFepsibias/out_2')
addpath('./param_fit_PF_KFpriorbias_empPrior/out_7')

% define model combinations
modstr      = {'prior','epsi'};
nmod        = numel(modstr);

% organize data
out_fits = struct;
elbos   = nan(2,4,2,nsubj); % rep/alt, quarter, model, subj
lls     = nan(2,4,2,nsubj);

for im = 1:2
    % extract data from subject data files
    jsubj = 1;
    for isubj = subjlist
        % load fit output
        if im == 1 
            % priorbias
            filename = sprintf('out_fit_KFempPrior_tsu_28_%02d',isubj);
        else
            % epsibias
            filename = sprintf('out_fit_KFepsibias_theta_biasSubj_tsu_28_%02d',isubj);
        end
        load(filename);
        % store model evidence
        for ic = 1:3
            for iq = 1:4
                elbos(ic,iq,im,jsubj)   = out_fit{ic,iq,isubj}.elbo;
                %lls(ic,iq,im,jsubj)     = out_fit{ic,iq,isubj}.l;
            end
        end
        jsubj = jsubj + 1;
    end
    out_fits(im).elbos  = elbos(:,:,im,:);
    out_fits(im).lls    = lls(:,:,im,:);
end

%% Fixed-effects Bayesian model selection

elbos_subj_desc = nan(nmod,nsubj,4,2);
i_elbos_subj    = nan(nmod,nsubj,4,2);

% winning model count for each quarter and condition for each subject
figure(1)
for ic = 1:3
    for iq = 1:4
        % rank models
        [elbos_subj_desc(:,:,iq,ic),i_elbos_subj(:,:,iq,ic)] = sort(reshape(elbos(ic,iq,:,:),[nmod,nsubj]),1,'descend');
        % plot
        subplot(3,4,4*(ic-1)+iq)
        histHandle = histogram(i_elbos_subj(1,:,iq,ic),[1:9],'FaceColor',graded_rgb(ic,iq));
        histHandle.BinEdges = histHandle.BinEdges - histHandle.BinWidth/2;
        hold on
        xticks(1:2)
        xlim([0 3])
        xticklabels(ticklabels)
    end
end
sgtitle(sprintf('Winning model counts for each condition and quarter\nFixed effects BMS'))

% calculate Bayes factors
bayesfacs = nan(nmod,nsubj,4,3); % model, subject, quarter, condition
for ic = 1:2
    for iq = 1:4
        for isubj = 1:nsubj
            iwinner = i_elbos_subj(1,isubj,iq,ic); % need to fix
            
            bayesfacs(iwinner,isubj,iq,ic) = 1;
            for imod = setdiff(1:nmod,iwinner)
                bayesfacs(imod,isubj,iq,ic) = exp(elbos(ic,iq,iwinner,isubj)-elbos(ic,iq,imod,isubj));
            end
        end
    end
end

% plot Bayes factor heatmap for specific condition and 
% > press the up/down arrows to change conditions
% > press the left/right arrows to change quarters
ic_p = 1;
iq_p = 1;
figure(2)
while true
    hm = heatmap(log10(bayesfacs(:,:,iq_p,ic_p)));
    colorbar
    colormap([linspace(1,.4,4)' linspace(1,.4,4)' ones(4,1)]);
    ylabel('Competing biased KF models');
    hm.YDisplayLabels = convertCharsToStrings(modstr);
    caxis([0 2]);
    xlabel('Subject');
    title(sprintf('Model comparison\ncondition: %d \nquarter: %d \n Log_10 of Bayes Factor',ic_p,iq_p))
    
    keypress = waitforbuttonpress;
    key = get(gcf,'CurrentKey');
    if strcmpi(num2str(key),'')
    elseif ismember(num2str(key),{'uparrow','downarrow'})
        ic_p = mod(ic_p,2)+1;
    elseif strcmpi(num2str(key),'leftarrow')
        iq_p = abs(mod(iq_p-2,4)+1);
    elseif strcmpi(num2str(key),'rightarrow')
        iq_p = abs(mod(iq_p,4)+1);
    else
    end
end

%% Random-effects Bayesian Model Selection
addpath('../Toolboxes/spm12')
alpha   = nan(1,nmod,3,4);
pexp    = nan(1,nmod,3,4);
pexc    = nan(1,nmod,3,4);

figure(3)
for ic = 1:3
    for iq = 1:4
        [alpha(:,:,ic,iq),pexp(:,:,ic,iq),pexc(:,:,ic,iq)] = spm_BMS(squeeze(elbos(ic,iq,:,:))');
        
        subplot(3,4,4*(ic-1)+iq)
        bar(1:nmod,pexc(:,:,ic,iq),'FaceColor',graded_rgb(ic,iq),'EdgeColor',[1 1 1]);
        ticklabels = {'prior','epsi'};
        xticklabels(ticklabels)
    end
end
sgtitle(sprintf('Exceedance probabilities of each model for each condition and quarter\nRandom effects BMS'))

% > The priorbias towards subjects' first response with argmax choice wins almost
%   all quarters in both the Repeating and Alternating conditions. 
% > The epsibias toward subjects' first response with argmax choice wins in the
%   3rd and 4th quarters in the Repeating condition.
% > Investigate qualitative measures in the check_fit_batch_KFbiased script.


%% Local functions
function rgb = graded_rgb(ic,iq)
           
    red = [1.0 .92 .92; .98 .80 .80; .97 .64 .64; .96 .49 .49];
    gre = [.94 .99 .94; .85 .95 .83; .74 .91 .70; .63 .87 .58];
    blu = [.93 .94 .98; .78 .85 .94; .61 .74 .89; .44 .63 .84];
           
    rgb = cat(3,red,gre);
    rgb = cat(3,rgb,blu);
    
    rgb = rgb(iq,:,ic);
end



