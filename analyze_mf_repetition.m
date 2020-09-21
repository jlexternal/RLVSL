% analyze_mf_repetition
%
% Objective: Analyze measures on blocks where subjects repeat their first
%               choice during the entirety of the block.
%
% Jun Seok Lee <jlexternal@gmail.com>



clear all;
close all;
addpath('./Toolboxes');

% Load subject data
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
% Load experiment structure
nsubj = numel(subjlist);
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',2,2);
load(filename,'expe');
nt = expe(1).cfg.ntrls;
nb = expe(1).cfg.nbout;
ntrain = expe(1).cfg.ntrain;
nb_c = nb/3;

idx_rep = nan(nb_c,3,nsubj); % index of blocks with complete repetition (general)
idx_inc = nan(nb_c,3,nsubj); % index of blocks with complete repetition (incorrect)

rts_blk = nan(nb_c,3,nsubj); % mean block RTs

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj);
    load(filename,'expe');
    ib_c = ones(3,1);
    for ib = 1+ntrain:nb+ntrain
        ctype = expe(ib).type;
        switch ctype
            case 'rnd' % random across successive blocks
                ic = 3;
            case 'alt' % always alternating
                ic = 2;
            case 'rep' % always the same
                ic = 1;
        end
        
        % organize responses
        resp = expe(ib).resp;
        resp(resp==2)=0;
        % check for absolute repetition during the block
        idx_rep(ib_c(ic),ic,jsubj) = isequal(ones(1,nt)*resp(1),resp);
        if idx_rep(ib_c(ic),ic,jsubj) == 1 && resp(1) ~= 1
            idx_inc(ib_c(ic),ic,jsubj) = 1;
        else
            idx_inc(ib_c(ic),ic,jsubj) = 0;
        end
        
        % organize reaction times
        rt = expe(ib).rt;
        rts_blk(ib_c(ic),ic,jsubj) = mean(rt);
        
        
        
        ib_c(ic) = ib_c(ic)+1;
    end
end

%% Calculate percentages of total repetition given quarter given condition
rep_percs = nan(4,3,nsubj);

for isubj = 1:nsubj
    for iq = 1:4
        for ic = 1:3
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            rep_percs(iq,ic,isubj) = sum(idx_rep(blockrange,ic,isubj))/4;
        end
    end
end

ngroups = size(rep_percs, 1);
nbars = size(rep_percs, 2);
colororder = [1 0 0; 0 1 0; 0 0 1];

rep_percs_m = mean(rep_percs,3);
bplot = bar(rep_percs_m);
for i = 1:3
    bplot(i).FaceColor = colororder(i,:);
end
hold on;
er = std(rep_percs,1,3)/sqrt(nsubj);
% Calculate the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
% Set the position of each error bar in the center of the main bar
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, rep_percs_m(:,i), er(:,i), 'k', 'linestyle', 'none');
end
xlabel('Quarter');
ylabel('Proportion');
title(sprintf('Proportion of absolute repetitive choice blocks\nError bars SEM'));
hold off

%% Calculate mean RTs on blocks split by absolute repetition 

rts_qt_split = nan(4,3,nsubj,2); % (qt,cond,subj,split)
for isubj = 1:nsubj
    for iq = 1:4
        for ic = 1:3
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            % on blocks of absolute repetition
            rts_qt_split(iq,ic,isubj,1) = mean(rts_blk(ismember(1:16,blockrange) & idx_rep(:,ic,isubj)',ic,isubj));
            % not on blocks of absolute repetition
            rts_qt_split(iq,ic,isubj,2) = mean(rts_blk(ismember(1:16,blockrange) & ~idx_rep(:,ic,isubj)',ic,isubj));
        end
    end
end

rts_qt_split_m = squeeze(mean(rts_qt_split(:,:,:,:),3,'omitnan')); % 4 qts x 3 conds x 2 splits

% calculate SEM separately for each quarter x condition (unequal number of
% subjects split on any given q x c)
figure;
ictr = 1;
er_rts = nan(4,3,2);
includeall = false; % false: remove subjects w/o split
for ic = 1:3
    for iq = 1:4
        for i = 1:2
            idx_nan = isnan(rts_qt_split(iq,ic,:,i));
            nsubj_sem = nsubj-sum(idx_nan);
            er_rts(iq,ic,i) = std(rts_qt_split(iq,ic,:,i),1,3,'omitnan')/sqrt(nsubj_sem);
        end
        % new variable to pass into Welch's test
        split1 = squeeze(rts_qt_split(iq,ic,:,1));
        split2 = squeeze(rts_qt_split(iq,ic,:,2));
        
        if ~includeall
            idx_nnan = bsxfun(@or,~isnan(split1),~~isnan(split2));
            split1 = split1(idx_nnan);
            split2 = split2(idx_nnan);
        end
        
        x = ones(size(split1));
        subplot(3,4,ictr)
        scatter(x*1,split1,'filled','MarkerEdgeAlpha',0,'MarkerFaceAlpha',.4);
        hold on;
        scatter(x*2,split2,'filled','MarkerEdgeAlpha',0,'MarkerFaceAlpha',.4);
        for i = 1:length(x)
            plot([1 2],[split1(i) split2(i)],'k:');
        end
        xticks([1 2]);
        xticklabels({'rep','not'});
        xlim([.5 2.5]);
        ylim([.2 1.7]);
        % calculate statistics within quarter on conditions (removing
        % subjects who do not have both)
        if includeall
            [h,p] = ttest2(split1,split2,'Vartype','unequal','Tail','left'); % Welch's t-test
        else
            [h,p] = ttest(split1,split2,'Tail','left'); % paired t-test
        end
        if h == 1
            text(1.5,1.5,'*','FontSize',20);
        end
        hold off;
        
        ictr = ictr + 1;
    end
end

% Plot mean reaction times over quarters x conditions split on blocks of absolute repetition
figure;
for ic = 1:3
    condrgb = zeros(1,3);
    condrgb(ic) = 1;
    subplot(3,1,ic);
    hold on;
    scatter([1:4]+.1*(ic-2),rts_qt_split_m(:,ic,1),'MarkerEdgeColor',condrgb,'HandleVisibility','off') % rep split
    plot([1:4]+.1*(ic-2),rts_qt_split_m(:,ic,1),'LineWidth',2,'Color',condrgb);
    errorbar([1:4]+.1*(ic-2),rts_qt_split_m(:,ic,1),er_rts(:,ic,1),'LineStyle','none','Color',condrgb,'HandleVisibility','off');
    
    scatter([1:4]+.1*(ic-2),rts_qt_split_m(:,ic,2),'x','MarkerEdgeColor',condrgb,'HandleVisibility','off') % not-rep split
    plot([1:4]+.1*(ic-2),rts_qt_split_m(:,ic,2),':','LineWidth',2,'Color',condrgb);
    errorbar([1:4]+.1*(ic-2),rts_qt_split_m(:,ic,2),er_rts(:,ic,2),'LineStyle','none','Color',condrgb,'HandleVisibility','off')
    legend({'Complete rep ','not'});
    xticks([1:4]);
    xlim([.5 4.5]);
    ylim([.5 .9])
    xlabel('Quarter');
    ylabel('Reaction time (s)');
    hold off;
end


%% Check RTs on reactions to disconfirmatory evidence on split 






