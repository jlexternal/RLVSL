% fit_mf_rlvsl.m

clear all;
close all;
ifig = 1;
nsubjtot    = 31;
excluded    = [1];
subjlist    = setdiff(1:nsubjtot, excluded);
subparsubjs = [excluded 11 23 28];
subjlist = setdiff(1:nsubjtot, subparsubjs); % if excluding underperforming/people who didn't get it
% load experiment structure
nsubj = numel(subjlist);
% Data manip step
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',2,2);
load(filename,'expe');
nt = expe(1).cfg.ntrls; % number of trials in a block
nb = expe(1).cfg.nbout; % number of total blocks 
nc = 3; % number of conditions
nq = 4; % number of quarters
ntrain = expe(1).cfg.ntrain;
nb_c = nb/nc; % number of blocks per condition

blcks       = nan(nb_c,nt,3,nsubj);
resps       = nan(nb_c,nt,3,nsubj);

mu_new   = 55;  % mean of higher distribution
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

a = sig_opti/expe(1).cfg.sgen;      % slope of linear transformation aX+b
b = mu_new - a*expe(1).cfg.mgen;    % intercept of linear transf. aX+b

for isubj = subjlist
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

        resp_mult = -(expe(ib).resp-1.5)*2;
        
        blcks(ib_c(ic),:,ic,isubj) = round(resp_mult.*expe(ib).blck*a+b);
        resps(ib_c(ic),:,ic,isubj) = -(expe(ib).resp-2);
        
        ib_c(ic) = ib_c(ic)+1;
    end
end

ind_subj_quarterly_acc_mean = zeros(numel(subjlist),nt,3,4); % isubj, it, ic, iquarter
quarterly_acc_mean  = zeros(4,nt,3); % iquarter, it, ic
quarterly_acc_sem   = zeros(4,nt,3); % iquarter, it, ic

% filter by quarter
for iq = 1:4
    for ic = 1:3
        blockindex = 4*(iq-1)+1:4*iq;
        for is = 1:numel(subjlist)
            ind_subj_quarterly_acc_mean(is,:,ic,iq) = mean(resps(blockindex,:,ic,subjlist(is)));
        end
        quarterly_acc_mean(iq,:,ic) = mean(ind_subj_quarterly_acc_mean(:,:,ic,iq));
        quarterly_acc_sem(iq,:,ic) = std(ind_subj_quarterly_acc_mean(:,:,ic,iq))/sqrt(numel(subjlist));
    end
end

%% Fit exponential functions to data (jackknifed resampling method)
out_fit = struct;
st_fit  = struct;
jk_fit  = struct;
bs_fit  = struct;
x0 = [1 .5 3];          % initial parameters
lb = [0 0 -Inf];        
ub = [1 1 Inf];
% parameters: x(1) = asymptote value
%             x(2) = "1-prior" parameter 
%             x(3) = time constant

xdatamax = size(ind_subj_quarterly_acc_mean,2);
xdata    = 0:xdatamax-1;

out_fit.cond = struct;
st_fit.cond  = struct;
jk_fit.cond  = struct;
bs_fit.cond  = struct;
for ic = 1:nc
    out_fit.cond(ic).quarter = struct;
    for iq = 1:4
        ydata = mean(ind_subj_quarterly_acc_mean(:,:,ic,iq),1);
        F = @(x) sum((x(1)-x(2)*exp(-(xdata)/x(3)) - ydata).^2);
        
        % standard procedure
        options = optimoptions('fmincon','Display','none');
        [x,fval,exitflag,output] = fmincon(F,x0,[],[],[],[],lb,ub,[],options);
        
        out_fit.cond(ic).quarter(iq).as = x(1);
        out_fit.cond(ic).quarter(iq).pr = x(2);
        out_fit.cond(ic).quarter(iq).tc = x(3);

        % visualize fits on the entire dataset 
        if false
            figure;
            hold on;
            scatter(xdata,ydata);
            plot(xdata,x(1)-exp(-(xdata+x(2))/x(3)));
            ylim([0 1]);
            hold off;
            pause;
        end
        
        
        % standard individual subj fit procedure
        stndfit = true;
        if stndfit 
            for isubj = 1:nsubj
                ydata = mean(ind_subj_quarterly_acc_mean(isubj,:,ic,iq),1);
                F = @(x) sum((x(1)-x(2)*exp(-(xdata)/x(3)) - ydata).^2);
                [x,fval,exitflag,output] = fmincon(F,x0,[],[],[],[],lb,ub);

                st_fit.cond(ic).quarter(iq).as(isubj) = x(1);
                st_fit.cond(ic).quarter(iq).pr(isubj) = x(2);
                st_fit.cond(ic).quarter(iq).tc(isubj) = x(3);
            end
        end
        
        % jackknife procedure
        jackknifing = false;
        if jackknifing
            for jsubj = 1:nsubj
                subjs = setdiff(1:nsubj,jsubj);
                ydata = mean(ind_subj_quarterly_acc_mean(subjs,:,ic,iq),1);
                F = @(x) sum((x(1)-x(2)*exp(-(xdata)/x(3)) - ydata).^2);

                [x,fval,exitflag,output] = fmincon(F,x0,[],[],[],[],lb,ub);

                jk_fit.cond(ic).quarter(iq).as(jsubj) = x(1);
                jk_fit.cond(ic).quarter(iq).pr(jsubj) = x(2);
                jk_fit.cond(ic).quarter(iq).tc(jsubj) = x(3);
            end
            % Jackknifed means
            out_fit.cond(ic).quarter(iq).as_mean_jk = mean(jk_fit.cond(ic).quarter(iq).as(:));
            out_fit.cond(ic).quarter(iq).pr_mean_jk = mean(jk_fit.cond(ic).quarter(iq).pr(:));
            out_fit.cond(ic).quarter(iq).tc_mean_jk = mean(jk_fit.cond(ic).quarter(iq).tc(:));
            % Jackknife-corrected variances
            out_fit.cond(ic).quarter(iq).as_var_jk = var(jk_fit.cond(ic).quarter(iq).as(:))*(nsubj-1);
            out_fit.cond(ic).quarter(iq).pr_var_jk = var(jk_fit.cond(ic).quarter(iq).pr(:))*(nsubj-1);
            out_fit.cond(ic).quarter(iq).tc_var_jk = var(jk_fit.cond(ic).quarter(iq).tc(:))*(nsubj-1);
        end
        
        % bootstrap procedure (not the main way) 
        bootstrapping = false;
        if bootstrapping
            nsamp_bs = 27; % number of samplings of the samplings
            for isamp = 1:nsamp_bs
                % sample n number of subjects from the subject pool with resampling
                subjs_bs = randsample(1:numel(subjlist),nsubj,true);
                ydata = mean(ind_subj_quarterly_acc_mean(subjs_bs,:,ic,iq),1);
                F = @(x) sum((x(1)-x(2)*exp(-(xdata)/x(3)) - ydata).^2);

                [x,fval,exitflag,output] = fmincon(F,x0,[],[],[],[],lb,ub);

                % log parameters
                bs_fit.cond(ic).quarter(iq).as(isamp) = x(1);
                bs_fit.cond(ic).quarter(iq).pr(isamp) = x(2);
                bs_fit.cond(ic).quarter(iq).tc(isamp) = x(3);
            end
        end
    end
end

%% Plot: Exponential fits by quarter (standard fit)
if stndfit
figure(ifig);
ifig = ifig + 1;
xs = 0:.25:15;
for ic = 1:3
    for iq = 1:4
        % calculate model curve limits
        expfit = zeros(1,numel(xs));
        for isubj = 1:nsubj
            as_st = st_fit.cond(ic).quarter(iq).as(isubj);
            pr_st = st_fit.cond(ic).quarter(iq).pr(isubj);
            tc_st = st_fit.cond(ic).quarter(iq).tc(isubj);
            expfit(isubj,:) = expfn(pr_st,tc_st,as_st,xs);
        end
        expfit_m = mean(expfit(:,:),1);
        
        %pause;
        expfit_var = var(expfit,0,1);
        
        subplot(1,4,iq);
        hold on;
        fig = errorbar([1:nt]+(.1*(ic-1)),quarterly_acc_mean(iq,:,ic),quarterly_acc_sem(iq,:,ic),'o',...
                       'Color',graded_rgb(ic,iq,4));
        fig.CapSize = 0; 
        fig.Marker = mrkr_type(ic);
        plot(xs+1, expfit_m,'Color',graded_rgb(ic,iq,4),'LineWidth',2);
        shadedErrorBar(xs+1,expfit_m,sqrt(expfit_var)/sqrt(nsubj),'lineprops',{'Color',graded_rgb(ic,iq,4)});
        xlim([1 16]);
        ylim([.2 1.05]);
    end
    hold off;
end
end
%% Plot: Exponential fits by condition (standard fit)
if stndfit
figure(ifig);
ifig = ifig + 1;
xs = 0:.25:15;
for ic = 1:3
    for iq = 1:4
        % calculate model curve limits
        expfit = zeros(1,numel(xs));
        for isubj = 1:nsubj
            as_st = st_fit.cond(ic).quarter(iq).as(isubj);
            pr_st = st_fit.cond(ic).quarter(iq).pr(isubj);
            tc_st = st_fit.cond(ic).quarter(iq).tc(isubj);
            expfit(isubj,:) = expfn(pr_st,tc_st,as_st,xs);
        end
        expfit_m = mean(expfit(:,:),1);
        
        %pause;
        expfit_var = var(expfit,0,1);
        
        subplot(3,1,ic);
        hold on;
        fig = errorbar([1:nt]+(.1*(ic-1)),quarterly_acc_mean(iq,:,ic),quarterly_acc_sem(iq,:,ic),'o',...
                       'Color',graded_rgb(ic,iq,4));
        fig.CapSize = 0; 
        fig.Marker = mrkr_type(ic);
        plot(xs+1, expfit_m,'Color',graded_rgb(ic,iq,4),'LineWidth',2);
        shadedErrorBar(xs+1,expfit_m,sqrt(expfit_var)/sqrt(nsubj),'lineprops',{'Color',graded_rgb(ic,iq,4)});
        yline(.5);
        xlim([1 16]);
        ylim([.3 1.05]);
    end
    hold off;
end
end
%% Plot: Exponential fits by quarter (jackknifed fit)

figure(ifig);
ifig = ifig + 1;
for ic = 1:3
    for iq = 1:4
        
        as = out_fit.cond(ic).quarter(iq).as_mean_jk;
        pr = out_fit.cond(ic).quarter(iq).pr_mean_jk;
        tc = out_fit.cond(ic).quarter(iq).tc_mean_jk;
        xs = 0:.25:15;
        expfit_m = expfn(pr,tc,as,xs);
        
        % calculate model curve limits
        expfit_jk = zeros(1,numel(xs));
        for isubj = 1:nsubj
            as_st = st_fit.cond(ic).quarter(iq).as(isubj);
            pr_st = st_fit.cond(ic).quarter(iq).pr(isubj);
            tc_st = st_fit.cond(ic).quarter(iq).tc(isubj);
            expfit_jk(isubj,:) = expfn(pr_st,tc_st,as_st,xs);
        end
        
        %pause;
        expfit_var_jk = var(expfit_jk,0,1)*(nsubj-1);
        
        subplot(1,4,iq);
        hold on;
        fig = errorbar([1:nt]+(.1*(ic-1)),quarterly_acc_mean(iq,:,ic),quarterly_acc_sem(iq,:,ic),'o',...
                       'Color',graded_rgb(ic,iq,4));
        fig.CapSize = 0; 
        fig.Marker = mrkr_type(ic);
        plot(xs+1, expfit_m,'Color',graded_rgb(ic,iq,4),'LineWidth',2);
        shadedErrorBar(xs+1,expfit_m,sqrt(expfit_var_jk),'lineprops',{'Color',graded_rgb(ic,iq,4)});
        ylim([.2 1.05]);
    end
    hold off;
end

%% Plot: Exponential fits by condition (jackknifed fit)
if jackknifing
figure(ifig);
ifig = ifig + 1;
for ic = 1:3
    for iq = 1:4
        as = out_fit.cond(ic).quarter(iq).as_mean_jk;
        pr = out_fit.cond(ic).quarter(iq).pr_mean_jk;
        tc = out_fit.cond(ic).quarter(iq).tc_mean_jk;
        xs = 0:.25:15;
        expfit_m = expfn(pr,tc,as,xs);
        
        % calculate model curve limits
        expfit_jk = zeros(1,numel(xs));
        for isubj = 1:nsubj
            as_jk = jk_fit.cond(ic).quarter(iq).as(isubj);
            pr_jk = jk_fit.cond(ic).quarter(iq).pr(isubj);
            tc_jk = jk_fit.cond(ic).quarter(iq).tc(isubj);
            
            expfit_jk(isubj,:) = expfn(pr_jk,tc_jk,as_jk,xs);
            %plot(xs,expfit_v(isubj,:));
        end
        
        %pause;
        expfit_jk = var(expfit_jk,0,1)*(nsubj-1);
    
        subplot(1,3,ic);
        hold on;
        fig = errorbar([1:nt]+(.1*(ic-1)),quarterly_acc_mean(iq,:,ic),quarterly_acc_sem(iq,:,ic),'o',...
                       'Color',graded_rgb(ic,iq,4));
        fig.CapSize = 0; 
        fig.Marker = mrkr_type(ic);
        plot(xs+1, expfit_m,'Color',graded_rgb(ic,iq,4),'LineWidth',2);
        shadedErrorBar(xs+1,expfit_m,sqrt(expfit_jk),'lineprops',{'Color',graded_rgb(ic,iq,4)});
        ylim([.2 1.05]);
    end
    hold off;
end
end

%% Plot: Exponential fit parameters by condition 
if jackknifing
as_jk = nan(nsubj,nq,nc);
pr_jk = nan(nsubj,nq,nc);
tc_jk = nan(nsubj,nq,nc);
as_bs = nan(nsamp_bs,nq,nc);
pr_bs = nan(nsamp_bs,nq,nc);
tc_bs = nan(nsamp_bs,nq,nc);
figure(ifig);
ifig = ifig + 1;
hold on;
for ic = 1:3
    for iq = 1:4
        for isubj = 1:nsubj
            as_jk(isubj,iq,ic) = jk_fit.cond(ic).quarter(iq).as(isubj);
            pr_jk(isubj,iq,ic) = 1-jk_fit.cond(ic).quarter(iq).pr(isubj);
            tc_jk(isubj,iq,ic) = jk_fit.cond(ic).quarter(iq).tc(isubj);
        end
        
        for isamp = 1:nsamp_bs
            as_bs(isamp,iq,ic) = bs_fit.cond(ic).quarter(iq).as(isamp);
            pr_bs(isamp,iq,ic) = 1-bs_fit.cond(ic).quarter(iq).pr(isamp);
            tc_bs(isamp,iq,ic) = bs_fit.cond(ic).quarter(iq).tc(isamp);
        end
    end
    subplot(1,4,1); % prior
    hold on;
    plot(1:nq,mean(pr_bs(:,:,ic)),':','LineWidth',2,'Color', graded_rgb(ic,iq,4)-.2);
    shadedErrorBar(1:nq,mean(pr_bs(:,:,ic)),sqrt(var(pr_bs(:,:,ic))*(nsubj/(nsubj-1))),...
                        'lineprops',{'Color',graded_rgb(ic,iq,4)-.2},'patchSaturation',.1);
    plot(1:nq,mean(pr_jk(:,:,ic)),'LineWidth',2,'Color', graded_rgb(ic,iq,4));
    shadedErrorBar(1:nq,mean(pr_jk(:,:,ic)),sqrt(var(pr_jk(:,:,ic))*(nsubj-1)),...
                        'lineprops',{'Color',graded_rgb(ic,iq,4)},'patchSaturation',.1);
    ylabel('Parameter value (arbitrary units)');
    title('"prior" param.');
    
    subplot(1,4,2); % time constant
    hold on;
    plot(1:nq,mean(tc_bs(:,:,ic)),':','LineWidth',2,'Color', graded_rgb(ic,iq,4)-.2);
    shadedErrorBar(1:nq,mean(tc_bs(:,:,ic)),sqrt(var(tc_bs(:,:,ic))*(nsubj/(nsubj-1))),...
                        'lineprops',{'Color',graded_rgb(ic,iq,4)-.2},'patchSaturation',.1);
    plot(1:nq,mean(tc_jk(:,:,ic)),'LineWidth',2,'Color', graded_rgb(ic,iq,4));
    shadedErrorBar(1:nq,mean(tc_jk(:,:,ic)),sqrt(var(tc_jk(:,:,ic))*(nsubj-1)),...
                        'lineprops',{'Color',graded_rgb(ic,iq,4)},'patchSaturation',.1);
    xlabel('Quarter');
    title('Time constant param.');
    
    subplot(1,4,3); % asymptote
    hold on;
    plot(1:nq,mean(as_bs(:,:,ic)),':','LineWidth',2,'Color', graded_rgb(ic,iq,4)-.2);
    shadedErrorBar(1:nq,mean(as_bs(:,:,ic)),sqrt(var(as_bs(:,:,ic))*(nsubj/(nsubj-1))),...
                        'lineprops',{'Color',graded_rgb(ic,iq,4)-.2},'patchSaturation',.1);
    plot(1:nq,mean(as_jk(:,:,ic)),'LineWidth',2,'Color', graded_rgb(ic,iq,4));
    shadedErrorBar(1:nq,mean(as_jk(:,:,ic)),sqrt(var(as_jk(:,:,ic))*(nsubj-1)),...
                        'lineprops',{'Color',graded_rgb(ic,iq,4)},'patchSaturation',.1);
    title('Asymptote param.');
    hold off;
     
    subplot(1,4,4);
    hold on;
    %plot(1:nq,mean(1./tc_bs(:,:,ic)),':','LineWidth',2,'Color', graded_rgb(ic,iq,4)-.2);
    %shadedErrorBar(1:nq,mean(1./tc_bs(:,:,ic)),sqrt(var(1./tc_bs(:,:,ic))*(nsubj/(nsubj-1))),...
    %                    'lineprops',{'Color',graded_rgb(ic,iq,4)-.2},'patchSaturation',.1);
    plot(1:nq,mean(1./tc_jk(:,:,ic)),'LineWidth',2,'Color', graded_rgb(ic,iq,4));
    shadedErrorBar(1:nq,mean(1./tc_jk(:,:,ic)),sqrt(var(1./tc_jk(:,:,ic))*(nsubj-1)),...
                        'lineprops',{'Color',graded_rgb(ic,iq,4)},'patchSaturation',.1);
    title('Learning rate (1/tau)');

end
end
%% Study on the exponential function's parameters
x = [0:.1:16];
% alter 'as' (asymptote)
figure(ifig);
ifig = ifig + 1;
subplot(1,4,1);
plot(x,expfn(.5,1,.7,x));
hold on;
plot(x,expfn(.5,1,.8,x));
plot(x,expfn(.5,1,1,x));
xlim([0 nt]);
ylim([0 1]);
hold off;
% => asymptote parameter bounds = [0 1] without interactive effects from other
%                                       parameters

% alter 'pr' (prior)
subplot(1,4,2);
plot(x,expfn(.5,1,1,x));
hold on;
plot(x,expfn(.4,1,1,x));
plot(x,expfn(.3,1,1,x));
xlim([0 nt]);
ylim([0 1]);
hold off;
% all other parameters constant, pr alters the initial value

% alter 'tc'
subplot(1,4,3);
plot(x,expfn(.5,2,1,x));
hold on;
plot(x,expfn(.5,1.5,1,x));
plot(x,expfn(.5,1,1,x));
plot(x,expfn(.5,.5,1,x));
xlim([0 nt]);
ylim([0 1]);
hold off
% all other parameters constant, tc alters the initial value as well as the rate of rise 
% decreasing values => higher rate of rise

% alter multiple to see interactions between parameters
subplot(1,4,4);
plot(x,expfn(0,1,1,x),'LineWidth',2);
hold on;
plot(x,expfn(1,1,1,x));
plot(x,expfn(0,.8,1,x));
plot(x,expfn(1,.8,1,x),'.');
xlim([1 nt]);
ylim([0 1]);
hold off;

%% Local Functions
function out = expfn(pr,tc,as,x)
    out = as-pr*exp(-x/tc);
end

function rgb = graded_rgb(ic,ib,nb)
    xc = linspace(.8,.4,nb);

    rgb =  [1,xc(ib),xc(ib); ...
               xc(ib),1,xc(ib); ...
               xc(ib),xc(ib),1];

    rgb = rgb(ic,:);
end

function mrkr = mrkr_type(ic)
    if ic == 3
        mrkr = '*';
    elseif ic == 2
        mrkr = '^';
    else 
        mrkr = 'o';
    end
end
