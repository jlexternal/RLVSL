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

out_fit_all = cell(3,4,nsubj);
params      = nan(3,4,nsubj,5); % cond,time,subj,param

bounds = [0 0; 0 0; 0 5; 0 10];

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    
    % load new fit file
    filename = sprintf('./param_fit_PF_KFepsibias/out_2/out_fit_KFepsibias_theta_biasSubj_tsu_%d_%02d.mat',nsubj,isubj);
    load(filename);
    
    for ic = 1:3
        for iq = 1:4
            out_fit_all{ic,iq,jsubj} = out_fit{ic,iq,isubj};
            for ip = 1:5
                params(ic,iq,jsubj,ip) = out_fit{ic,iq,isubj}.xmap(ip);
            end
        end
    end
end
clearvars out_fit

%% plot parameter evolution over time
parstr = {'kini','kinf','zeta','theta','epsi'};
figure
hold on
for ip = 1:5
    subplot(5,1,ip);
    for ic = 1:3
        errorbar([1:4]+(ic-2)*.1,mean(params(ic,1:4,:,ip),3),std(params(ic,1:4,:,ip),1,3),...
                        'o','Color',graded_rgb(ic,4),'LineWidth',2,'LineStyle','none');
        hold on
    end
    xticks(1:4)
    xticklabels(1:4)
    xlim([0.5 4.5]);
    title(sprintf(parstr{ip}));
end


%% check certain correlations between parameters
ipar1 = 5;
ipar2 = 1;

for ipar2 = 1:4
    titletxt = sprintf('Correlation, %s vs %s',parstr{ipar2},parstr{ipar1});

    par1 = [];
    par2 = [];
    for icond = 1:3
        for itime = 1:4
            par1 = [par1 squeeze(params(icond,itime,:,ipar1))'];
            par2 = [par2 squeeze(params(icond,itime,:,ipar2))'];
        end
    end
    figure(ipar2)
    istart = 1;
    for icond = 1:3
        if ismember(icond,1:2)
            scatter(par1(istart:istart+112-1),par2(istart:istart+112-1),'MarkerFaceColor',graded_rgb(icond,4),'MarkerEdgeColor',graded_rgb(icond,4))
            istart = istart+112;
        else
            scatter(par1(istart:end),par2(istart:end),'MarkerFaceColor',graded_rgb(icond,4),'MarkerEdgeColor',graded_rgb(icond,4))
        end
        hold on
    end
    %plot(0:1,polyval(polyfit(par1,par2,1),0:1));

    % remove hgihest values of epsi (as it overrides effect of all other parameters)
    idx = par1<.9;
    plot(0:1,polyval(polyfit(par1(idx),par2(idx),1),0:1));

    [r,p] = corrcoef(par1(idx),par2(idx));
    text(.5,.5,['r=' num2str(r(1,2))]);
    text(.5,.45,['p=' num2str(p(1,2))]);
    if ipar1 == 5 && ~isempty(idx)
        xline(.9,':');
    end

    title(titletxt);
    xlabel(parstr{ipar1});
    ylabel(parstr{ipar2});
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