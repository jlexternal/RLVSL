% check_fit_batch_noisyKF_empPrior

clear all
clc
addpath('..');
%% Data organization
% gather all out_fit structures into 1 file
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

out_fit_all = cell(3,5,nsubj);
params      = nan(3,5,nsubj,4);

oldpars = load('out_fit_noisyKF');
pars_old = nan(3,5,nsubj,4);

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    
    % load file
    filename = sprintf('./param_fit_PF_KFpriorbias_empPrior/out/out_fit_KFempPrior_tsu_%d_%02d.mat',nsubj,isubj);
    load(filename);
    
    for ic = 1:3
        if ic ~= 3
            for iq = 1:4
                out_fit_all{ic,iq,jsubj} = out_fit{ic,iq,isubj};
                for ip = 1:4
                    if ip < 3
                        params(ic,iq,jsubj,ip) = 1/(1+exp(-out_fit{ic,iq,isubj}.xmap(ip)));
                    else
                        params(ic,iq,jsubj,ip) = exp(out_fit{ic,iq,isubj}.xmap(ip));
                    end
                    pars_old(ic,iq,jsubj,ip) = oldpars.out_fit_all{ic,iq,jsubj}.xmap(ip);
                end
            end
        else
            out_fit_all{ic,5,jsubj} = out_fit{ic,5,isubj};
            for ip = 1:4
                if ip < 3
                    params(ic,5,jsubj,ip) = 1/(1+exp(-out_fit{ic,5,isubj}.xmap(ip)));
                else
                    params(ic,5,jsubj,ip) = exp(out_fit{ic,5,isubj}.xmap(ip));
                end
                pars_old(ic,5,jsubj,ip) = oldpars.out_fit_all{ic,5,jsubj}.xmap(ip);
            end
        end
    end
end
clearvars out_fit


%% plot parameter evolution over time
parstr = {'kini','kinf','zeta','theta'};
figure
hold on
for ip = 1:4
    subplot(4,1,ip);
    for ic = 1:2
        errorbar(1:4,mean(params(ic,1:4,:,ip),3),std(params(ic,1:4,:,ip),1,3),...
                        'o','Color',graded_rgb(ic,4),'LineWidth',2,'LineStyle','none');
        hold on
        errorbar([1:4]-.1,mean(pars_old(ic,1:4,:,ip),3),std(pars_old(ic,1:4,:,ip),1,3),...
                        'x','Color',graded_rgb(ic,2),'LineWidth',1,'LineStyle','none');
    end
    ic = 3;
    shadedErrorBar([1 4],mean(params(3,5,:,ip),3)*ones(1,2),std(params(3,5,:,ip),1,3)*ones(1,2),...
                    'lineprops',{'Color',graded_rgb(ic,4)},'patchSaturation',.3);
    shadedErrorBar([1 4]-.1,mean(pars_old(3,5,:,ip),3)*ones(1,2),std(pars_old(3,5,:,ip),1,3)*ones(1,2),...
                    'lineprops',{':','Color',graded_rgb(ic,2)},'patchSaturation',.3);
    xticks(1:4)
    xticklabels(1:4)
    title(sprintf(parstr{ip}));
end

%%
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