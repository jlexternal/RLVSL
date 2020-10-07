% run_model_selection_batch_KFbiased
%
% Objective: Compare the different fitting options for the unbiased Kalman filter
%             model on the random condition and choose which set of options best fit the
%             subjects' data.
%
% Jun Seok Lee <jlexternal@gmail.com>
clc
% loop through and store in local memory
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

% unbiased KF fitting options
cscheme = 'ths';
lscheme = 'sym';
nscheme = 'upd';

load('subj_resp_rew_all.mat'); % load all relevant subject data
addpath('./param_fit_PF_KFepsibias')
addpath('./param_fit_PF_KFpriorbias')

% define model space
biastyps = {'epsi','prio'}; % bias incorporation type
biassrcs = {'corr','subj'}; % bias source
choicplc = {'argm','soft'}; % choice/selection policy

% define model combinations
vec_combis  = combvec(1:length(biastyps),1:length(biassrcs),1:length(choicplc));
nmod        = size(vec_combis,2);
modstr      = cell(1,nmod);

% organize data
out_fits = struct;
elbos   = nan(2,4,nmod,nsubj); % rep/alt, quarter, model, subj
lls     = nan(2,4,nmod,nsubj);

for im = 1:nmod
    % define models
    out_fits(im).biastype     = biastyps{vec_combis(1,im)};
    out_fits(im).biassource   = biassrcs{vec_combis(2,im)};
    out_fits(im).selpolicy    = choicplc{vec_combis(3,im)};
    % set strings for folder and file search
    if strcmpi(biastyps{vec_combis(1,im)},'epsi')
        bt_str = 'epsi';
    else
        bt_str = 'prior';
    end
    if strcmpi(biassrcs{vec_combis(2,im)},'corr')
        bc_str = 'Corr';
    else
        bc_str = 'Subj';
    end
    if strcmpi(choicplc{vec_combis(3,im)},'soft')
        sn_str = 'theta_';
    else
        sn_str = '';
    end
    % extract data from subject data files
    modstr{im} = sprintf('%s_%s_%s',biastyps{vec_combis(1,im)},biassrcs{vec_combis(2,im)},choicplc{vec_combis(3,im)});
    jsubj = 1;
    for isubj = subjlist
        % load fit output
        filename = sprintf('./param_fit_PF_KF%sbias/out/out_fit_KF%sbias_%sbias%s_%s%s%s_%d_%02d.mat',...
                       bt_str,bt_str,sn_str,bc_str,...
                       cscheme(1),lscheme(1),nscheme(1),nsubj,isubj);
        load(filename);
        % store model evidence
        for ic = 1:2
            for iq = 1:4
                elbos(ic,iq,im,jsubj)   = out_fit{ic,iq,isubj}.elbo;
                lls(ic,iq,im,jsubj)     = out_fit{ic,iq,isubj}.l;
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
for ic = 1:2
    for iq = 1:4
        % rank models
        [elbos_subj_desc(:,:,iq,ic),i_elbos_subj(:,:,iq,ic)] = sort(reshape(elbos(ic,iq,:,:),[nmod,nsubj]),1,'descend');
        % plot
        subplot(2,4,4*(ic-1)+iq)
        histHandle = histogram(i_elbos_subj(1,:,iq,ic),[1:9]);
        histHandle.BinEdges = histHandle.BinEdges - histHandle.BinWidth/2;
        hold on
        ticklabels = sprintf('%s\\newline%s\\newline%s\n',...
                    [convertCharsToStrings(extractBetween(modstr,1,4));...
                     convertCharsToStrings(extractBetween(modstr,6,9));...
                     convertCharsToStrings(extractBetween(modstr,11,14))]);
        xticks(1:8)
        xlim([0 9])
        xticklabels(ticklabels)
        title('Winning model counts for each condition and quarter')
        hold off
    end
end

% calculate Bayes factors
bayesfacs = nan(nmod,nsubj,4,2);
for ic = 1:2
    for iq = 1:4
        for isubj = 1:nsubj
            iwinner = i_elbos_subj(1,isubj,iq,ic);
            
            bayesfacs(iwinner,isubj,iq,ic) = 1;
            for imod = setdiff(1:nmod,iwinner)
                bayesfacs(imod,isubj,iq,ic) = exp(elbos(ic,iq,iwinner,isubj)-elbos(ic,iq,imod,isubj));
            end
        end
    end
end

% plot Bayes factor heatmap for specific condition and quarter
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
alpha   = nan(1,nmod,2,4);
pexp    = nan(1,nmod,2,4);
pexc    = nan(1,nmod,2,4);

figure(3)
for ic = 1:2
    for iq = 1:4
        [alpha(:,:,ic,iq),pexp(:,:,ic,iq),pexc(:,:,ic,iq)] = spm_BMS(squeeze(elbos(ic,iq,:,:))');
        
        subplot(2,4,4*(ic-1)+iq)
        bar(1:nmod,pexc(:,:,ic,iq),'FaceColor',graded_rgb(ic,iq),'EdgeColor',[0 0 0]);
        ticklabels = sprintf('%s\\newline%s\\newline%s\n',...
                    [convertCharsToStrings(extractBetween(modstr,1,4));...
                     convertCharsToStrings(extractBetween(modstr,6,9));...
                     convertCharsToStrings(extractBetween(modstr,11,14))]);
        xticks(1:8)
        xlim([0 9])
        xticklabels(ticklabels)
    end
end
sgtitle('Exceedance probabilities of each model for each condition and quarter')


%% Local functions
function rgb = graded_rgb(ic,iq)
           
    red = [1.0 .92 .92; .98 .80 .80; .97 .64 .64; .96 .49 .49];
    gre = [.94 .99 .94; .85 .95 .83; .74 .91 .70; .63 .87 .58];
    blu = [.93 .94 .98; .78 .85 .94; .61 .74 .89; .44 .63 .84];
           
    rgb = cat(3,red,gre);
    rgb = cat(3,rgb,blu);
    
    rgb = rgb(iq,:,ic);
end



