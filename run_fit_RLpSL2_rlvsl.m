% run_fit_RLpSL2_rlvsl

clear all
close all
clc

% add VBMC toolbox to path
addpath('./vbmc');

% conditions to fit
cond2fit = {'rnd'}; % specify which conditions; 'rep', 'alt', 'rnd', or 'all'
cond = [];
if ~ismember(cond2fit,{'all','rep','alt','rnd'})
    error('Unrecognisable condition!');
end
if strcmpi(cond2fit{1},'all')
    cond = 1:3;
end
if ismember(ismember(cond2fit,'rep'),1)
    cond = cat(1,cond,1);
end
if ismember(ismember(cond2fit,'alt'),1)
    cond = cat(1,cond,2);
end
if ismember(ismember(cond2fit,'rnd'),1)
    cond = cat(1,cond,3);
end

% quarters to fit
quar2fit = 1:4;

% subjects to be included in analysis
nsubjtot    = 31;
excluded    = [1];
subjlist    = setdiff(1:nsubjtot, excluded);
subparsubjs = [excluded 11 23 28];
subjlist = setdiff(1:nsubjtot, subparsubjs); % if excluding underperforming/people who didn't get it

nsubj       = numel(subjlist);
nb          = 16;
nt          = 16;

% organize response and outcome data structures for fitting function
cond_struct = struct;

for icond = cond
    choices = zeros(nb,nt,nsubj);
    outcome = zeros(nb,nt,nsubj);
    
    switch icond
        case 3 % random across successive blocks
            condtype = 'rnd';
        case 2 % always alternating
            condtype = 'alt';
        case 1 % always the same
            condtype = 'rep';
    end
    
    is = 0; % relative subj. indexing
    for isubj = subjlist % absolute subj. indexing
        filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)
        if ~exist(filename,'file')
            error('Missing experiment file!');
        end
        load(filename,'expe');
        is   = is + 1;
        
        ib_c = 0;
        for ib = 1:expe(1).cfg.nbout+3
            ctype = expe(ib).type;
            if ismember(ctype,condtype)
                ib_c = ib_c + 1;
                choices(ib_c,:,is) = expe(ib).resp;
                outcome(ib_c,:,is) = expe(ib).blck_trn;
            else
                
            end
        end
    end
    
    cond_struct(icond).outcome = outcome;
    cond_struct(icond).choices = choices;
end
clearvars outcome choices

%% Fit model for each subject based on condition (and quarter)

out_fit  = cell(nsubj,1);

% trial-to-trial RL difficulty calculation
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

for isubj = 1:nsubj
    
    out_fit{isubj}.cond = struct;
    
    for icond = cond
        % can consider a conditional for the 'rnd' case to analyze over the entire 16
        % blocks instead of by quarters
        
        blk = kron(1:nb,ones(1,nt))';
        trl = repmat((1:nt)',[nb,1]);
        
        resp = cond_struct(icond).choices(:,:,isubj)';
        resp = resp(:); % vectorize response matrix
        
        rew = nan(size(resp,1),2); % rewards of chosen & unchosen options
        for i = 1:size(resp,1) % pointers on blk and trl
            
            rew(i,resp(i))   = cond_struct(icond).outcome(blk(i),trl(i),isubj)/100; % chosen
            rew(i,3-resp(i)) = 1-rew(i,resp(i));                                    % unchosen
        end
        
        % instantiate configuration structure
        cfg         = [];
        cfg.icond   = icond; % condition type
        cfg.nsmp    = 1e3;   % number of samples
        cfg.verbose = true;  % verbose VBMC output
        cfg.stgt    = sig_opti/100;
        %cfg.prior = .5; %test
        
        % fixed parameters
        cfg.tau     = 1e-6; % assume argmax choice policy
        cfg.ksi     = 1e-6; % assume pure Weber noise (no constant term)
        
        % fit on quarters (in non-random conditions)
        for iq = quar2fit
            % to be fitted
                % zeta  - scaling of learning noise with prediction error
                % alpha - KF learning rate asymptote
                % prior - strength of the prior belief on the correct option
            fprintf('Fitting subj %d on condition %d over quarter %d\n',isubj,icond,iq);
            
            cfg.iquarter = iq;
            
            % chunk data structures into quarters
            blkstart  = 4*(iq-1)+1;
            blkend    = blkstart+3;
            
            idx     = ismember(blk,blkstart:blkend);    % blocks corresponding to the quarter
            
            qresp   = resp.*idx;
            qrew    = rew.*(idx.*ones(size(idx,1),2));
            qtrl    = trl.*idx;
            qresp   = nonzeros(qresp);
            qrew    = [nonzeros(qrew(:,1)) nonzeros(qrew(:,2))];
            qtrl    = nonzeros(qtrl);
            
            cfg.resp    = qresp;
            cfg.rew     = qrew;
            cfg.trl     = qtrl;
            
%           out_fit{isubj}.cond(icond).quarter(iq).fit = fit_noisyRL(cfg);
            out_fit{isubj}.cond(icond).quarter(iq).fit = fit_RLpSL2_rlvsl(cfg);
            
            rz = out_fit{isubj}.cond(icond).quarter(iq).fit.zeta;
            ra = out_fit{isubj}.cond(icond).quarter(iq).fit.alpha;
            rp = out_fit{isubj}.cond(icond).quarter(iq).fit.priorpar;
            fprintf('Found params: zeta: %.2f, alpha: %.2f, prior: %.2f\n',rz,ra,rp);
            
        end
        
    end
    
end

%% Visualize parameters

ncond = 3;
nquar = 4;
nsubj = numel(out_fit);
zetas  = zeros(ncond,nquar,nsubj);
alphas = zeros(ncond,nquar,nsubj);
priors = zeros(ncond,nquar,nsubj);


for isubj = 1:nsubj
    for icond = 1:ncond
        for iquar = 1:nquar
            zetas(icond,iquar,isubj)  = out_fit{isubj}.cond(icond).quarter(iquar).fit.zeta;
            alphas(icond,iquar,isubj) = out_fit{isubj}.cond(icond).quarter(iquar).fit.alpha;
            priors(icond,iquar,isubj) = out_fit{isubj}.cond(icond).quarter(iquar).fit.prior;
        end
    end
    
end

%% priors over time
close all;
clf;
for icond = 1:3
    figure(icond);
    hold on;
    for iquar = 1:nquar
        prplot = scatter(iquar*ones(1,nsubj),priors(icond,iquar,:),'MarkerEdgeColor',graded_rgb(icond,iquar,4));
        scatter(iquar, mean(priors(icond,iquar,:),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
        errorbar(iquar,mean(priors(icond,iquar,:),3),std(priors(icond,iquar,:)),'Color',graded_rgb(icond,iquar,4));
        ylim([0 1]);

        % check significance of increasing prior with ranksum test
    end
    for isubj = 1:nsubj
        for ix = 1:3
            plot([ix ix+1], priors(icond,ix:ix+1,isubj),'Color',[0 0 0]);
        end
    end
    hold off;
    
    figure(11);
    hold on;
    if icond < 4
        for iquar = 1:nquar
            scatter(iquar, mean(priors(icond,iquar,:),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
            errorbar(iquar,mean(priors(icond,iquar,:),3),std(priors(icond,iquar,:)),'Color',graded_rgb(icond,iquar,4));
        end
    end
    hold off;
end
%% zetas over time
for icond = 1:ncond
    figure(icond+3);
    hold on;
    for iquar = 1:nquar
        scatter(iquar*ones(1,nsubj),zetas(icond,iquar,:),'MarkerEdgeColor',graded_rgb(icond,iquar,4));
        scatter(iquar, mean(zetas(icond,iquar,:),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
        errorbar(iquar,mean(zetas(icond,iquar,:),3),std(zetas(icond,iquar,:)),'Color',graded_rgb(icond,iquar,4));
        ylim([0 1]);

    end
    for isubj = 1:nsubj
        for ix = 1:3
            plot([ix ix+1], zetas(icond,ix:ix+1,isubj),'Color',[0 0 0]);
        end
    end
    hold off;
    
    figure(12);
    hold on;
    if icond < 4
        for iquar = 1:nquar
            scatter(iquar, mean(zetas(icond,iquar,:),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
            errorbar(iquar,mean(zetas(icond,iquar,:),3),std(zetas(icond,iquar,:)),'Color',graded_rgb(icond,iquar,4));
        end
    end
    hold off;
    
end

%% alphas over time
for icond = 1:ncond
    figure(icond+6);
    hold on;
    for iquar = 1:nquar
        scatter(iquar*ones(1,nsubj),alphas(icond,iquar,:),'MarkerEdgeColor',graded_rgb(icond,iquar,4));
        scatter(iquar, mean(alphas(icond,iquar,:),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
        errorbar(iquar,mean(alphas(icond,iquar,:),3),std(alphas(icond,iquar,:)),'Color',graded_rgb(icond,iquar,4));
        ylim([0 1]);

    end
    for isubj = 1:nsubj
        for ix = 1:3
            plot([ix ix+1], alphas(icond,ix:ix+1,isubj),'Color',[0 0 0]);
        end
    end
    hold off;
    
    figure(13);
    hold on;
    if icond < 4
        for iquar = 1:nquar
            scatter(iquar, mean(alphas(icond,iquar,:),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
            errorbar(iquar,mean(alphas(icond,iquar,:),3),std(alphas(icond,iquar,:)),'Color',graded_rgb(icond,iquar,4));
        end
    end
    hold off;
end


%%

function rgb = graded_rgb(ic,ib,nb)
    xc = linspace(.8,.2,nb);

    rgb =  [1,xc(ib),xc(ib); ...
               xc(ib),1,xc(ib); ...
               xc(ib),xc(ib),1];

    rgb = rgb(ic,:);
end