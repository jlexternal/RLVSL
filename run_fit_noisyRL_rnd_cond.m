
%%

% clear workspace
%clear all
%close all
%clc

% add VBMC toolbox to path
addpath('./vbmc');

%load('./rnd_cond.mat');

nblk  = size(rnd_cond.choices,1);
ntrl  = size(rnd_cond.choices,2);
nsubj = size(rnd_cond.choices,3);

out_fit = cell(nsubj,1);

for isubj = 1:nsubj
    
    
    blk = kron(1:nblk,ones(1,ntrl))';
    trl = repmat((1:ntrl)',[nblk,1]);
    
    resp = rnd_cond.choices(:,:,isubj)';
    resp = resp(:);
    
    rew = nan(size(resp,1),2);
    for i = 1:size(resp,1)
        rew(i,resp(i))   = rnd_cond.outcome(blk(i),trl(i),isubj)/100;
        rew(i,3-resp(i)) = 1-rew(i,resp(i));
    end
    
    % create configuration structure
    cfg         = [];
    cfg.resp    = resp; % response
    cfg.rew     = rew; % reward values
    cfg.trl     = trl; % trial number in current block
    
    cfg.fbtype  = 1; % partial feedback
    cfg.nsmp    = 1e3; % number of samples
    cfg.verbose = true; % verbose VBMC output?
    cfg.alphau  = nan; % assume policy learning
    cfg.ksi     = 1e-6; % assume pure Weber noise (no constant term)
    cfg.tau     = 1e-6; % assume argmax choice policy
    
    %out_fit{isubj} = fit_noisyRL(cfg);
    out_fit{isubj} = fit_noisyRL_commented(cfg);
    
end

%%

for isubj = 1:nsubj
    clf;
    Xs = vbmc_rnd(out_fit{isubj}.vp{1},1e6);  % Generate samples from the variational posterior

    % We compute the pdf of the approximate posterior on a 2-D grid
    plot_lb = [0 0];
    plot_ub = quantile(Xs,0.999);
    x1 = linspace(plot_lb(1),plot_ub(1),400);
    x2 = linspace(plot_lb(2),plot_ub(2),400);
    [xa,xb] = meshgrid(x1,x2);  % Build the grid
    xx = [xa(:),xb(:)];         % Convert grids to a vertical array of 2-D points
    yy = real(vbmc_pdf(out_fit{isubj}.vp{1},xx));       % Compute PDF values on specified points

    % Plot approximate posterior pdf (works only in 1-D and 2-D)
    surf(x1,x2,reshape(yy,[numel(x1),numel(x2)]),'EdgeColor','none');
    xlabel('alphac');
    ylabel('zeta');
    zlabel('Approximate posterior pdf');
    set(gca,'TickDir','out');
    set(gcf,'Color','w');
    pause;
end
