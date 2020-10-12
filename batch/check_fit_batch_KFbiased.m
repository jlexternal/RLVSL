% check_fit_batch_KFbiased
%
% Objective: Check fitted parameters for the biased KF model 
%
% Version:   Code for fits from the fit_noisyKF_epsibias.m fitting code (pre 12 Oct 2020)
%
% Jun Seok Lee <jlexternal@gmail.com>

clear all
clc

iscompare = true;

if ~iscompare
    iscompare = false;
    nmod = 1;
end

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);
addpath('./param_fit_PF_KFepsibias')
addpath('./param_fit_PF_KFpriorbias')
addpath('../')
load('subj_resp_rew_all.mat'); % load all relevant subject data

% fitting options (fix to 'ths', 'sym', 'upd' after KFunbiased analysis)
cscheme = 'ths';
lscheme = 'sym';
nscheme = 'upd';

bt_comp = {'epsi','prior'};
bc_comp = [false true];
sn_comp = [false true];

% ------------------------ model options -------------------------
% (if comparing models, the values set here will be the ones that remain fixed)
biastype = 'epsi'; % 'epsi'        or  'prior'
biascorr = false;   % true: correct     false: subject 1st choice
selnoise = false;   % true: softmax     false: argmax
% ---------------------- comparison options -----------------------
if iscompare
    % toggle the model options to be compared
    iscomp_bt = true;
    iscomp_bc = false;
    iscomp_sn = false;
    
    nmod = prod([iscomp_bt iscomp_bc iscomp_sn]+1);
    vec_combis = combvec(0:double(iscomp_bt),0:double(iscomp_bc),0:double(iscomp_sn));
end
% -----------------------------------------------------------------
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
params      = squeeze(nan(npars,nsubjtot,ncond,4,nmod)); % param, subj, cond, qtr
params_rnd  = nan(3,nsubjtot); % parameters from the unbiased KF fits to the random condition

% load and gather all the fit data
out_fit_all = squeeze(cell(ncond,4,nsubjtot,nmod));

for imod = 1:nmod
    if iscompare
        if iscomp_bt
            biastype = bt_comp{vec_combis(1,imod)+1};
        end
        if iscomp_bc
            biascorr = logical(vec_combis(2,imod));
        end
        if iscomp_sn
            selnoise = logical(vec_combis(3,imod));
        end
    end
    
    for isubj = subjlist
        filename = sprintf('./param_fit_PF_KF%sbias/out/out_fit_KF%sbias_%sbias%s_%s%s%s_%d_%02d.mat',...
                           biastype,biastype,sn_str,bc_str,...
                           cscheme(1),lscheme(1),nscheme(1),nsubj,isubj);
        load(filename);
        for ic = 1:2
            for iq = 1:4
                out_fit_all{ic,iq,isubj,imod} = out_fit{ic,iq,isubj};
                params(1,isubj,ic,iq,imod) = out_fit{ic,iq,isubj}.kini;
                params(2,isubj,ic,iq,imod) = out_fit{ic,iq,isubj}.kinf;
                params(3,isubj,ic,iq,imod) = out_fit{ic,iq,isubj}.zeta;
                params(4,isubj,ic,iq,imod) = out_fit{ic,iq,isubj}.epsi;
                params(5,isubj,ic,iq,imod) = out_fit{ic,iq,isubj}.theta;
            end
        end
        % get unbiased KF fit parameters for comparison
        
        if ~exist('isrnddone','var')
            isrnddone = false;
        end
        if ~isrnddone
            filename = sprintf('./param_fit_PF_KFunbiased/options_fit/out_fit_KFunbiased_%s_%s_%s_%d_%02d.mat',...
                               cscheme,lscheme,nscheme,nsubj,isubj);
            load(filename);
            params_rnd(1,isubj) = out_fit{3,isubj}.kini;
            params_rnd(2,isubj) = out_fit{3,isubj}.kinf;
            params_rnd(3,isubj) = out_fit{3,isubj}.zeta;
            if isubj == subjlist(end)
                isrnddone = true;
            end
        end
    end
end

% ----------- define conditions and quarters ----------- 
cond_sim = [1];
quar_sim = [1:4];
% ------------------------------------------------------
%% Visualize evolution of fitted parameters over time in each condition
condstr = {'Repeating','Alternating'};
parstr  = {'kini','kinf','zeta','epsi','theta'}; 
for imod = 1:nmod
    figure
    for ic = 1
        subplot(length(cond_sim),1,ic)
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
                        params(ip,subjlist,ic,iq,imod),...
                        80,param_rgb(ip),'filled','MarkerFaceAlpha',.1,'HandleVisibility','off')

                errorbar(iq+((ip-3)*.1),mean(params(ip,subjlist,ic,iq,imod)),...
                            std(params(ip,subjlist,ic,iq,imod),1,2),'o',...
                            'MarkerEdgeColor',param_rgb(ip),'MarkerFaceColor',param_rgb(ip),...
                            'Color',param_rgb(ip),'LineWidth',2,'CapSize',0,'HandleVisibility',handlestr)
            end
            if ismember(ip,[1 2 3])
                % connecting lines for each subject
                plot(x,squeeze(params(ip,subjlist,ic,:,imod))','Color',[param_rgb(ip) .05],'HandleVisibility','off')
                % connecting lines for the mean parameters
                plot([1:4]+((ip-3)*.1),squeeze(mean(params(ip,subjlist,ic,:,imod),2))','Color',param_rgb(ip),...
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
                        'Choice policy: %s'],bt_comp{imod},bc_str,sn_p_str))
    end
end

%% Simulate models with fitted parameters for given set of conditions and quarters

sim_out = cell(max(cond_sim),max(quar_sim),nsubjtot,nmod);
for imod = 1:nmod
    % experimental parameters
    cfg = struct;
    cfg.nb = 4; % nb is 4, not 16, since simulating quarters
    cfg.nt = 16;
    cfg.ms = .55;   cfg.vs = .07413^2; 
    cfg.sbias_cor = false;  
    cfg.sbias_ini = false;
    cfg.cscheme = cscheme;  cfg.lscheme = lscheme;  cfg.nscheme = nscheme;
    cfg.ns      = 1000; 
    cfg.ksi     = 0;
    cfg.sameexpe = true;    % true if all sims see the same reward scheme
    
    if iscompare
        if iscomp_bt
            biastype = bt_comp{vec_combis(1,imod)+1};
        end
        if iscomp_bc
            biascorr = logical(vec_combis(2,imod));
        end
        if iscomp_sn
            selnoise = logical(vec_combis(3,imod));
        end
    end
    if strcmpi(biastype,'prior')
        cfg.sbias_ini = true;
        cfg.epsi = 0;
    end
    if biascorr
        cfg.sbias_cor = true;
    end
    if ~selnoise
        cfg.theta = 0;
    end
    fprintf('Simulating model %d...\n',imod);
    % put fit parameters here and simulate
    for icond = cond_sim
        fprintf('on condition %d...\n',icond);
        for iq = quar_sim
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            for isubj = subjlist
                cfg.kini    = params(1,isubj,ic,iq,imod);
                cfg.kinf    = params(2,isubj,ic,iq,imod);
                cfg.zeta    = params(3,isubj,ic,iq,imod);
                cfg.epsi    = params(4,isubj,ic,iq,imod);
                cfg.theta   = params(5,isubj,ic,iq,imod);
                
                cfg.firstresp = subj_resp_rew_all(isubj).resp(blockrange,1,icond); % simulations make the same 1st choice as subject
                cfg.compexpe  = subj_resp_rew_all(isubj).rew_expe(blockrange,:,icond)/100;
                
                sim_out{icond,iq,isubj,imod} = sim_epsibias_fn(cfg);
            end
        end
    end
end

%% Calculate model-free measures
% 1/ Learning curves

lcurve= nan(4,cfg.nt,ncond,nsubj,nmod+1); % quarter, trial, condition, subject, source(subj+nmods)

for isubj = subjlist
    jsubj = find(subjlist==isubj);
    
    for icond = cond_sim
        for iq = quar_sim
            % subjects
            resp = subj_resp_rew_all(isubj).resp;
            resp(resp==2)=0;
            blockrange = 4*(iq-1)+1:4*(iq-1)+4;
            lcurve(iq,:,icond,jsubj,1) = mean(resp(blockrange,:,icond),1); % subj curve
            
            % simulations
            for imod = 1:nmod
                resp = sim_out{icond,iq,isubj,imod}.resp;
                resp(resp==2) = 0;
                lcurve(iq,:,icond,jsubj,imod+1) = mean(mean(resp(:,:,:),1),3);
            end
        end
    end
end

%% calculate learning curves
leg_ctr = 1;
mod_lstyle = {'--',':','-o','-x'};
figure
for ic = cond_sim
    hold on
    for iq = quar_sim
        subplot(1,length(quar_sim),find(quar_sim==iq))
        errorbar(1:16,mean(lcurve(iq,:,ic,:,1),4),std(lcurve(iq,:,ic,:,1),1,4)/sqrt(nsubj),...
            'CapSize',0,'Color',graded_rgb(ic,iq),'LineWidth',1.5)
        legtxt{leg_ctr} = sprintf('Subjects');
        leg_ctr = leg_ctr + 1;
        for imod = 1:nmod
            shadedErrorBar(1:16,mean(lcurve(iq,:,ic,:,imod+1),4),std(lcurve(iq,:,ic,:,2),1,4)/sqrt(nsubj),...
                'lineprops',{mod_lstyle{imod},'Color',graded_rgb(ic,iq),'LineWidth',1.5},'patchSaturation',.1)
            legtxt{leg_ctr} = sprintf('Model %d sim',imod);
            leg_ctr = leg_ctr + 1;
        end
        ylim([.4 1]);
        legend(legtxt,'Location','southeast')
        title(sprintf('Condition: %d, Quarter: %d',ic,iq))
    end
    hold off
end
sgtitle(sprintf('Learning curves\nError bars SEM'))


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