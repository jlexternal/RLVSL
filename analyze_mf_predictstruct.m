% analyze_mf_predictstruct
%
% Objective: Investigate how much predictive power the last choice in a previous
%            block has on the first choice in the subsequent one for the structured conditions
%            in experiment RLVSL
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

% 
resps = nan(nb_c,nt,3,nsubj);

disp('Loading subject data...');
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
        resp = expe(ib).resp;
        resps(ib_c(ic),:,ic,jsubj) = resp;
        
        ib_c(ic) = ib_c(ic)+1;
    end
end

%% Calculate proportion of subjects whose 1st choice of a block corresponded structurally with the last choice in the prev block

resp_1st = nan(nb_c-1,3,nsubj); %block, condition, subj

for isubj = 1:nsubj
    for ic = 1:3
        resp_1st(:,ic,isubj) = bsxfun(@eq,resps(1:nb_c-1,end,ic,isubj),resps(2:nb_c,1,ic,isubj))';
    end
end

% Explanation of the figure:
%   Repeating:      if the curve rises, the subjects are repeating their choices from the previous block, whether correct or not
%   Alternating:    if the curve rises, the subjects are alternating from their previous choice from the last block, correct or not
%   Random:         doesn't mean anything really as there are no shapes in common w/ the previous blocks
figure;
hold on;
colorRGB = [1 0 0; 0 1 0; 0 0 1];
for ic = 1:3
    shadedErrorBar(2:nb_c,mean(resp_1st(:,ic,:),3),std(resp_1st(:,ic,:),1,3)/sqrt(nsubj),'lineprops',{'LineWidth',2,'Color',colorRGB(ic,:)},'patchSaturation',.1);
    scatter(2:nb_c,mean(resp_1st(:,ic,:),3),std(resp_1st(:,ic,:),1,3)/sqrt(nsubj))
    l = lsline;
end
yline(.5,':');
yline(1);
xlabel('Block number');
ylabel('Proportion');
title('Proportion of (in)correct 1st choice on block i given last choice on block i-1 was (in)correct');



