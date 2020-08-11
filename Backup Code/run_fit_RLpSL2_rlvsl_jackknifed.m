% run_fit_RLpSL2_rlvsl_test

clear all
close all
clc

% add VBMC toolbox to path
addpath('./vbmc');

% conditions to fit
cond2fit = {'all'}; % specify which conditions; 'rep', 'alt', 'rnd', or 'all'
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
ncond = 3;
nquar = 4;

out_params = cell(ncond,nquar);

% trial-to-trial RL difficulty calculation
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

for icond = cond
    for iquar = quar2fit
        out_params{icond,iquar} = struct;
        
        % instantiate configuration structure
        cfg         = [];
        cfg.icond   = icond; % condition type
        cfg.nsmp    = 1e3;   % number of samples
        cfg.verbose = true;  % verbose VBMC output
        cfg.stgt    = sig_opti/100;

        % fixed parameters
        cfg.tau     = 1e-6; % assume argmax choice policy
        cfg.ksi     = 1e-6; % assume pure Weber noise (no constant term)
        
        fprintf('Jackknife fitting over subjects on condition %d over quarter %d\n',icond,iquar);
        for esubj = 1:nsubj % esubj - subj left out

            blk = kron(1:nb,ones(1,nt))';
            trl = repmat((1:nt)',[nb,1]);
            blk = repmat(blk,[nsubj-1 1]);
            trl = repmat(trl,[nsubj-1 1]);
            
            subjs = setdiff([1:nsubj],esubj);

            % Vectorize responses and rewards
            resp = cond_struct(icond).choices(:,:,subjs); 
            resp = num2cell(resp,[1 2]); 
            resp = vertcat(resp{:})'; % stack responses, and transpose for vectorisation
            resp = resp(:); % vectorise response matrix
            
            rews = cond_struct(icond).outcome(:,:,subjs)/100;
            rews = num2cell(rews,[1 2]);
            rews = vertcat(rews{:})';
            rews = rews(:);
            
            rew = nan(size(resp,1),2); % rewards of chosen & unchosen options
            for i = 1:size(resp,1) 
                rew(i,resp(i))   = rews(i);     % chosen
                rew(i,3-resp(i)) = 1-rews(i);   % unchosen
            end
        
            % to be fitted
                % zeta  - scaling of learning noise with prediction error
                % alpha - KF learning rate asymptote
                % prior - strength of the prior belief on the correct option
            cfg.iquarter = iquar;

            % chunk data structures into quarters
            blkstart  = 4*(iquar-1)+1;
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

            out_params{icond,iquar}.fit(esubj).fit = fit_RLpSL2_rlvsl(cfg);

            % store fitted parameters to structure
            out_params{icond,iquar}.zeta(esubj)   = out_params{icond,iquar}.fit(esubj).fit.zeta;
            out_params{icond,iquar}.alpha(esubj)  = out_params{icond,iquar}.fit(esubj).fit.alpha;
            out_params{icond,iquar}.prior(esubj)  = out_params{icond,iquar}.fit(esubj).fit.prior;

        end
        rz = mean(out_params{icond,iquar}.zeta(:));
        ra = mean(out_params{icond,iquar}.alpha(:));
        rp = mean(out_params{icond,iquar}.prior(:));
        rz_v = var(out_params{icond,iquar}.zeta(:))*(nsubj-1);
        ra_v = var(out_params{icond,iquar}.alpha(:))*(nsubj-1);
        rp_v = var(out_params{icond,iquar}.prior(:))*(nsubj-1);
        fprintf('Jackknife fits on condition %d over quarter %d\n',icond,iquar);
        fprintf('Mean params: zeta: %.2f, alpha: %.2f, prior: %.2f\n',rz,ra,rp);
        fprintf('Variance:    zeta: %.2f, alpha: %.2f, prior: %.2f\n',rz_v,ra_v,rp_v);
    end
end

%% Visualize parameters

ncond = 3;
nquar = 4;
nsubj = numel(subjlist);
zetas  = zeros(ncond,nquar);
alphas = zeros(ncond,nquar);
priors = zeros(ncond,nquar);
zetas_v  = zeros(ncond,nquar);
alphas_v = zeros(ncond,nquar);
priors_v = zeros(ncond,nquar);


for icond = 1:ncond
    for iquar = 1:nquar
        zetas(icond,iquar)  = mean(out_params{icond,iquar}.zeta);
        alphas(icond,iquar) = mean(out_params{icond,iquar}.alpha);
        priors(icond,iquar) = mean(out_params{icond,iquar}.prior);
        zetas_v(icond,iquar)  = var(out_params{icond,iquar}.zeta)*(nsubj-1);
        alphas_v(icond,iquar) = var(out_params{icond,iquar}.alpha)*(nsubj-1);
        priors_v(icond,iquar) = var(out_params{icond,iquar}.prior)*(nsubj-1);
    end
end


%% priors over time
close all;
clf;
figure(1);
for icond = 1:3
    hold on;
    if icond < 4
        for iquar = 1:nquar
            scatter(iquar, mean(priors(icond,iquar),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
            errorbar(iquar,mean(priors(icond,iquar),3),sqrt(priors_v(icond,iquar)),'Color',graded_rgb(icond,iquar,4));
        end
    end
end
title(sprintf('Jackknifed prior parameter; n=%d',nsubj));
hold off;
%% zetas over time
figure(2);
for icond = 1:ncond
    hold on;
    if icond < 4
        for iquar = 1:nquar
            scatter(iquar, mean(zetas(icond,iquar),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
            errorbar(iquar,mean(zetas(icond,iquar),3),sqrt(zetas_v(icond,iquar)),'Color',graded_rgb(icond,iquar,4));
        end
    end
end
title(sprintf('Jackknifed learning noise parameter; n=%d',nsubj));
hold off;
%% alphas over time
figure(3);
for icond = 1:ncond
    hold on;
    if icond < 4
        for iquar = 1:nquar
            scatter(iquar, mean(alphas(icond,iquar),3),200,'filled','MarkerFaceColor',graded_rgb(icond,iquar,4));
            errorbar(iquar,mean(alphas(icond,iquar),3),sqrt(alphas_v(icond,iquar)),'Color',graded_rgb(icond,iquar,4));
        end
    end
end
title(sprintf('Jackknifed learning rate asymptote parameter; n=%d',nsubj));
hold off;
%%

function rgb = graded_rgb(ic,ib,nb)
    xc = linspace(.8,.2,nb);

    rgb =  [1,xc(ib),xc(ib); ...
               xc(ib),1,xc(ib); ...
               xc(ib),xc(ib),1];

    rgb = rgb(ic,:);
end