% analyze_mf_biascluster
%
% Objective: Validate the use of the epsilon-bias parameter in a model-free way by
%               calculating the cumulative accuracy curves and fitting a logistic
%               function
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
p_cor = nan(nb_c,1,3,nsubj);
p_1st = nan(nb_c,1,3,nsubj);
p_rep = nan(nb_c,1,3,nsubj);

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj);
    fprintf('Loading subject %d...\n',isubj);
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
        resp(resp==2)=0;
        
        p_cor(ib_c(ic),1,ic,jsubj) = sum(resp)/nt;
        p_1st(ib_c(ic),1,ic,jsubj) = numel(resp(resp == resp(1)))/nt;
        p_rep(ib_c(ic),1,ic,jsubj) = sum(bsxfun(@eq,resp(1:nt-1),resp(2:nt)))/nt;
        
        ib_c(ic) = ib_c(ic)+1;
    end
end

%% Visualize all
colors = ['r','g','b'];
mtype = ['o','x','+'];
condstr = {'Repeating','Alternating','Random'};
nbins = 10;
test1 = [];
test2 = [];
test3 = [];
figure;
for ic = 1:3 
    for isubj = 1:nsubj
        test1 = cat(1,test1,p_1st(:,1,ic,isubj));
        test2 = cat(1,test2,p_cor(:,1,ic,isubj));
        test3 = cat(1,test3,p_rep(:,1,ic,isubj));
    end
    
end

subplot(1,3,1);
hist3([test1 test2],[nbins nbins],'CDataMode','auto');
xlabel('p(1st response)');
ylabel('p(correct)');
title('correct x 1st response');

subplot(1,3,2);
hist3([test1 test3],[nbins nbins],'CDataMode','auto');
xlabel('p(1st response)');
ylabel('p(repeat)');
title('repeat x 1st response');

subplot(1,3,3);
hist3([test2 test3],[nbins nbins],'CDataMode','auto');
xlabel('p(correct)');
ylabel('p(repeat)');
title('repeat x correct');

% clustering
idx = kmeans([test1 test2 test3],2);

% scatter plot
figure;
jitter = normrnd(0,.01,[numel(test1) 3]);
for i = 1:2
    scatter3(test1(idx==i)+jitter(idx==i,1),test2(idx==i)+jitter(idx==i,2),test3(idx==i)+jitter(idx==i,3),'.');
    hold on;
end
xlabel('p(1st response)');
ylabel('p(correct)');
zlabel('p(repeat)');
%}
%% Visualize by condition
clf;
colors = ['r','g','b'];
mtype = ['o','x','+'];
condstr = {'Repeating','Alternating','Random'};
nbins = 10;
for ic = 1:3 
    test1 = [];
    test2 = [];
    test3 = [];
    for isubj = 1:nsubj
        test1 = cat(1,test1,p_1st(:,1,ic,isubj));
        test2 = cat(1,test2,p_cor(:,1,ic,isubj));
        test3 = cat(1,test3,p_rep(:,1,ic,isubj));
    end
    subplot(3,3,3*(ic-1)+1);
    histogram2(test1, test2,[nbins nbins],'Normalization','probability','ShowEmptyBins','on','FaceColor','flat');
    xlabel('p(1st response)');
    ylabel('p(correct)');
    title(sprintf('%s: correct x 1st response',condstr{ic}));

    subplot(3,3,3*(ic-1)+2);
    histogram2(test1, test3,[nbins nbins],'Normalization','probability','ShowEmptyBins','on','FaceColor','flat');
    xlabel('p(1st response)');
    ylabel('p(repeat)');
    title(sprintf('%s: repeat x 1st response',condstr{ic}));

    subplot(3,3,3*(ic-1)+3);
    histogram2(test2, test3,[nbins nbins],'Normalization','probability','ShowEmptyBins','on','FaceColor','flat');
    xlabel('p(correct)');
    ylabel('p(repeat)');
    title(sprintf('%s: repeat x correct',condstr{ic}));
end
