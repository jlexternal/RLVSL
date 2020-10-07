% check_fit_batch_KFbiased
%
% Objective: Check fitted parameters for the biased KF model 
%
% Jun Seok Lee <jlexternal@gmail.com>
clear all
clc

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);
addpath('./param_fit_PF_KFepsibias')
addpath('./param_fit_PF_KFpriorbias')
addpath('../')
load('subj_resp_rew_all.mat'); % load all relevant subject data

% fitting options
cscheme = 'ths';
lscheme = 'sym';
nscheme = 'upd';

% -------- model options ---------
biastype = 'prior';  % 'epsi' or 'prior'
biascorr = false;    % true: correct     false: subject 1st choice
selnoise = false;   % true: softmax     false: argmax
% ---------------------------------
npars = 5; % kini, kinf, zeta, (epsi), (theta)
ncond = 2;

if ~ismember(biastype,{'epsi','prior'})
    error('Unknown bias type! Indicate either ''epsi'' or ''prior'' as the biastype.')
end
if biascorr
    bc_str = 'Corr';
else
    bc_str = 'Subj';
end
if selnoise
    sn_str = 'theta_';
    sn_p_str = 'softmax';
else
    sn_str = '';
    sn_p_str = 'argmax';
end
% store fit parameters from each model
%   1/kini
%   2/kinf
%   3/zeta
%   4/epsi  if epsilon model
%   5/theta if softmax temperature is fit
params      = nan(npars,nsubjtot,ncond,4); % param, subj, cond, qtr
params_rnd  = nan(3,nsubjtot);

% load and gather all the fit data
out_fit_all = cell(ncond,4,nsubjtot);
for isubj = subjlist
    filename = sprintf('./param_fit_PF_KF%sbias/out/out_fit_KF%sbias_%sbias%s_%s%s%s_%d_%02d.mat',...
                       biastype,biastype,sn_str,bc_str,...
                       cscheme(1),lscheme(1),nscheme(1),nsubj,isubj);
    load(filename);
    for ic = 1:2
        for iq = 1:4
            out_fit_all{ic,iq,isubj} = out_fit{ic,iq,isubj};
            params(1,isubj,ic,iq) = out_fit{ic,iq,isubj}.kini;
            params(2,isubj,ic,iq) = out_fit{ic,iq,isubj}.kinf;
            params(3,isubj,ic,iq) = out_fit{ic,iq,isubj}.zeta;
            params(4,isubj,ic,iq) = out_fit{ic,iq,isubj}.epsi;
            params(5,isubj,ic,iq) = out_fit{ic,iq,isubj}.theta;
        end
    end
    % get unbiased KF fit parameters for comparison
    filename = sprintf('./param_fit_PF_KFunbiased/options_fit/out_fit_KFunbiased_%s_%s_%s_%d_%02d.mat',...
                       cscheme,lscheme,nscheme,nsubj,isubj);
    load(filename);
    params_rnd(1,isubj) = out_fit{3,isubj}.kini;
    params_rnd(2,isubj) = out_fit{3,isubj}.kinf;
    params_rnd(3,isubj) = out_fit{3,isubj}.zeta;
end

%% Visualize evolution of fitted parameters over time in each condition
figure(99)
condstr = {'Repeating','Alternating'};
parstr  = {'kini','kinf','zeta','epsi','theta'}; 
for ic = 1:2
    subplot(2,1,ic)
    hold on
    yline(0,'--','Color',[.8 .8 .8],'HandleVisibility','off');
    yline(1,'--','Color',[.8 .8 .8],'HandleVisibility','off');
    for ip=1:size(params,1)
        for iq = 1:4
            if iq == 1
                handlestr = 'on';
            else
                handlestr = 'off';
            end
            x(iq,:) = iq*ones(1,nsubj)+((ip-3)*.1)+normrnd(0,.01,[1,nsubj]);
            scatter(x(iq,:),...
                    params(ip,~isnan(params(ip,:,ic,iq)),ic,iq),...
                    80,param_rgb(ip),'filled','MarkerFaceAlpha',.1,'HandleVisibility','off')
                
            errorbar(iq+((ip-3)*.1),mean(params(ip,subjlist,ic,iq)),...
                        std(params(ip,subjlist,ic,iq),1,2),'o',...
                        'MarkerEdgeColor',param_rgb(ip),'MarkerFaceColor',param_rgb(ip),...
                        'Color',param_rgb(ip),'LineWidth',2,'CapSize',0,'HandleVisibility',handlestr)
        end
        if ismember(ip,[1 2 3])
            % connecting lines for each subject
            plot(x,squeeze(params(ip,subjlist,ic,:))','Color',[param_rgb(ip) .05],'HandleVisibility','off')
            % connecting lines for the mean parameters
            plot([1:4]+((ip-3)*.1),squeeze(mean(params(ip,subjlist,ic,:),2))','Color',param_rgb(ip),...
                'LineWidth',2,'HandleVisibility','off')
        end
        % random/novel condition parameters
        for ip = 1:size(params_rnd,1)
            shadedErrorBar([0 5],mean(params_rnd(ip,:),'omitnan')*ones(1,2),std(params_rnd(ip,:),1,2,'omitnan')*ones(1,2),...
                            'lineprops',{'Color',param_rgb(ip),'HandleVisibility','off'},'patchSaturation',.03)
        end
    end
    if ic == 1
        legend(parstr)
    end
    xlim([.5 4.5])
    ylim([-.2 1.5])
    xticks(1:4)
    set(gca,'TickDir','out');
    xlabel('Quarter')
    ylabel('Parameter value')
    title(sprintf('Condition: %s',condstr{ic}),'FontSize',12)
    sgtitle(sprintf(['Fitted parameters from %sbias model\n Bias towards %s response\n'...
                    'Choice policy: %s'],biastype,bc_str,sn_p_str))
end


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
            95 77 44;...
            69 50 67]...
            /100;
    rgb = rgb(ip,:);
end