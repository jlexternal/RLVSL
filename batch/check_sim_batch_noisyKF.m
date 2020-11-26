% check_sim_batch_noisyKF

clear all
% Import data from batch-ran model simulations
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

dir_str = 'epsi';

for isubj = 1:nsubj
    % load file
    filename = sprintf('./sim_noisyKF_paramfit/out_%s/out_resp_sim_noisyKF_%02d_%02d.mat',dir_str,nsubj,isubj);
    sim_out = load(filename);
    
    if isubj == 1
        resp_sim = nan(4,16,size(sim_out.resp_sim,3),3,4,nsubj);
    end
    % transfer data
    resp_sim(:,:,:,:,:,isubj) = sim_out.resp_sim(:,:,:,:,:,isubj); % block, trial, nsims, condition, quarter, subject
end

load('subj_resp_rew_all')

%% plot learning curves

lcurve= nan(4,16,3,nsubj,2); % quarter, trial, condition, subject

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    for icond = 1:3
        for iq = 1:4
            % subjects
            resp = subj_resp_rew_all(isubj).resp;
            resp(resp==2)=0;
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            lcurve(iq,:,icond,jsubj,1) = mean(resp(blockrange,:,icond),1); % subj curve
            
            % simulations
            resp = resp_sim(:,:,:,icond,iq,jsubj);
            resp(resp==2) = 0;
            lcurve(iq,:,icond,jsubj,2) = mean(mean(resp(:,:,:),1),3);
        end
    end
end

% plot learning curves
figure
for ic = 1:3
    subplot(1,3,ic)
    for iq = 1:4
        errorbar(1:16,mean(lcurve(iq,:,ic,:,1),4),std(lcurve(iq,:,ic,:,1),1,4)/sqrt(nsubj),...
            'CapSize',0,'Color',graded_rgb(ic,iq),'LineWidth',1.5)
        hold on
        shadedErrorBar(1:16,mean(lcurve(iq,:,ic,:,2),4),std(lcurve(iq,:,ic,:,2),1,4)/sqrt(nsubj),...
            'lineprops',{'--','Color',graded_rgb(ic,iq),'LineWidth',1.5},'patchSaturation',.1)
        title(sprintf('Condition: %d',ic))
    end
    ylim([.4 1]);
    title(sprintf('Condition: %d',ic))
    hold off
end
sgtitle(sprintf('Learning curves\nError bars SEM\n param source: xmap'))

%% Local functions
function rgb = graded_rgb(ic,iq)
           
    red = [1.0 .92 .92; .98 .80 .80; .97 .64 .64; .96 .49 .49];
    gre = [.94 .99 .94; .85 .95 .83; .74 .91 .70; .63 .87 .58];
    blu = [.93 .94 .98; .78 .85 .94; .61 .74 .89; .44 .63 .84];
           
    rgb = cat(3,red,gre);
    rgb = cat(3,rgb,blu);
    
    rgb = rgb(iq,:,ic);
end

function rgb = param_rgb(ip)
    rgb = [ 40 36 36;...
            66 61 61;...
            52 77 91;...
            69 50 67]...
            /100;
    rgb = rgb(ip,:);
end