function [testblocks] = testcase_sim_blocks(cfg,n_blocks,nsims)
% TESTCASE SIM BLOCKS Generates testcase blocks based on desired level of accuracy
%                       performed by the KF 
%
% Jun Seok Lee - Oct 2019

% Theoretical generative distributions
%{
figure(1);
dist1 = plot([1:99], normpdf([1:99],center-d,stdev));
c{1} = get(dist1,'Color');
hold on;
xline(center-d,'--','Color',c{1},'LineWidth',2);
dist2 = plot([1:99], normpdf([1:99],center+d,stdev));
c{2} = get(dist2,'Color');
xline(center+d,'--','Color',c{2},'LineWidth',2);
xlim([0,100]);
hold off;
%}

% Generate blocks and simulate optimal responses
resps   = [];
percs   = [];
blocks  = [];
q       = [];

for imeans = 1:nsims
    [resp, blck, qs]    = testcase_gen_block(cfg); 
    resps               = resp(numel(resp)); % 1-lower mean, 2-higher mean
    percs(imeans)       = numel(resps(resps==2))/numel(resps);
    blocks(:,:,imeans)  = blck;
    q(imeans,:)         = qs;
end

fprintf('Out of %d simulations, the average rate of success is %.2f with stdev = %2.2f' ,nsims,mean(percs),std(percs));
% Above, "success" means to choose the option with higher distribution mean
disp(' ');

% Choose blocks for testing based on ranks of the moment differences for each block

% need q values to get mean / var rankings from optimal observer
q_means = mean(q,2);
q_vars  = var(q,0,2);
%disp(mean(q_means)); %debug
%disp(mean(q_vars)); %debug
diff_means_sq   = (q_means-mean(q_means)).^2;   % calculate squared diff. of mean q per block to overall mean q
diff_vars_sq    = (q_vars-mean(q_vars)).^2;     % calculate squared diff. of var q per block to overall mean var
% rank these two measures
[~,imeans]  = sort(diff_means_sq,'ascend');     % i___ - 'i' refers to index
[~,ivars]   = sort(diff_vars_sq,'ascend');
rank_means  = 1:length(diff_means_sq);
rank_vars   = 1:length(diff_vars_sq);
rank_means(imeans)  = rank_means; 
rank_vars(ivars)    = rank_vars;
% add the ranks (lower ranks will be chosen as test blocks)
rank_total = rank_means + rank_vars;

[~,iblocks] = sort(rank_total,'ascend'); % 'iblocks' contains indices of blocks with best ranking
% choose the top 10 most average blocks
iblocks = iblocks(1:n_blocks);

% put the most average blocks into the output structure
testblocks = [];
i = 1;
for iblock = iblocks
    testblocks(:,:,i) = blocks(:,:,iblock);
    i = i+1;
end

end