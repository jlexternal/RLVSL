% analyze_mf_biascluster_logistic
%
% Objective: Validate the use of the epsilon-bias parameter in a model-free way by
%               calculating the cumulative accuracy curves and fitting a logistic
%               function
%
% Jun Seok Lee <jlexternal@gmail.com>

clear all;
close all;
addpath('./Toolboxes');

% Load subject data
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
% Load experiment structure
nsubj = numel(subjlist);
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',2,2);
load(filename,'expe');
nt = expe(1).cfg.ntrls;
nb = expe(1).cfg.nbout;
ntrain = expe(1).cfg.ntrain;
nb_c = nb/3;

% 
cumacc      = nan(nb+3,nt,nsubj);   % matches the block numbering of the expe structure
cumacc_cnd  = nan(nb_c,nt,3,nsubj); % cumulative accuracy per block by condition
cumacc_qtr  = nan(4,nt,3,nsubj);    % mean cum acc for each quarter per condition

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
        resp = expe(ib).resp;
        resp(resp==2)=0;
        % (use ib to track the block to those in the expe structure)
        cumacc(ib,:,isubj) = cumsum(resp)/nt; % cumacc for block
        cumacc_cnd(ib_c(ic),:,ic,isubj) = cumacc(ib,:,isubj); % cumacc for block in condition
        
        iq = floor((ib_c(ic)-1e-3)/4)+1; % calculate current quarter
        if mod(ib_c(ic),4) == 0
            cumacc_qtr(iq,:,ic,isubj) =  mean(cumacc_cnd(ib_c(ic)-3:ib_c(ic),:,ic,isubj),1); % mean cumacc per quarter in condition
        end
        ib_c(ic) = ib_c(ic)+1;
    end
end

%% Fit logistic curves to cumulative accuracy curves (quarters x conditions)

pvs = cell(4,3);

pfit_ini = [8   ;1   ]; % initial values
pfit_min = [-16 ;1e-3]; % minimum values
pfit_max = [16  ;4   ]; % maximum values

logi    = @(p_in)1./(1+exp(-p_in(2).*([1:nt]-p_in(1)-8))); % logistic function
objperf = @(p_in)sum((logi(p_in)-[1:nt]/16).^2);

% find parameters for perfect performance
disp('Finding best fitting parameters for perfect performance...');
p_perf = fmincon(objperf,pfit_ini,[],[],[],[],pfit_min,pfit_max,[], ...
                 optimset('Display','notify','FunValCheck','on',    ...
                 'Algorithm','interior-point','TolX',1e-20,'MaxFunEvals',1e6));
             
obj1error = @(p_in)sum((logi(p_in)-[0 [1:nt-1]/16]).^2);
p_1error = fmincon(obj1error,pfit_ini,[],[],[],[],pfit_min,pfit_max,[], ...
                 optimset('Display','notify','FunValCheck','on',    ...
                 'Algorithm','interior-point','TolX',1e-20,'MaxFunEvals',1e6));

fit_struct = struct;
fit_struct(1).pvals_q = {};
disp('Finding best fitting parameters to subject data...');
for ic = 1:3
    for iq = 1:4
        subj_ctr = 1;
        pv_temp = [];
        for isubj = subjlist
            cumacc_i = cumacc_qtr(iq,:,ic,isubj);
            objfn = @(p_in)sum((logi(p_in)-cumacc_i).^2);
            
            % find best-fitting parameters for subject x condition x quarter
            fit_struct(isubj).pvals_q{iq,ic} = fmincon(objfn,pfit_ini,[],[],[],[],pfit_min,pfit_max,[], ...
                                                       optimset('Display','notify','FunValCheck','on',  ...
                                                       'Algorithm','interior-point','TolX',1e-20,'MaxFunEvals',1e6));
            % organize param values by condition and quarter
            pv_temp = cat(1,pv_temp,fit_struct(isubj).pvals_q{iq,ic}');
            subj_ctr = subj_ctr + 1;
        end
        pvs{iq,ic} = pv_temp;
    end
end

%% Visualize clustering and compare to quarters on epsi-bias split

% structure parameters for k-medoids or k-means clustering algo
iskmeans = true;
pv_all = []; % parameter values
for iq = 1:4
    for ic = 1:3
        pv_all = cat(1,pv_all,pvs{iq,ic});
    end
end
% run clustering algo
if iskmeans
    idx_k = kmeans([pv_all(:,1) pv_all(:,2)],2); % index on k-mean clusters
else
    idx_k = kmedoids([pv_all(:,1) pv_all(:,2)],2); % index on k-medoid clusters
end
pv1 = pv_all(:,1);
pv2 = pv_all(:,2);

% Load epsilon-bias percentages
load('out_fit_epsi.mat'); % (subj,cond,quarter,?)
epsi(:,:,:,1) = [];
idx_e = epsi>=.5;
idx_e = idx_e(:);

% Plot: rerun until the colors of the crosshair match the cluster it's on top of)
%   (May need to run a few times since the indices will represent biased/unbiased quarters
%   based on the k-*** algo random seed)
clf;
% epsilon-bias quarters
p_epsi = scatter(pv1(idx_e==1),pv2(idx_e==1),100,'r','filled');
p_epsi.MarkerFaceAlpha=.4;
hold on;
% clustered on model-free parameters
cl1 = scatter(pv1(idx_k==1),pv2(idx_k==1),'.','r'); % cluster 1
cl2 = scatter(pv1(idx_k==2),pv2(idx_k==2),'.','b'); % cluster 2
% statistics on cluster
xc = mean(pv1(idx_k==1));
yc = mean(pv2(idx_k==1));
xe = std(pv1(idx_k==1));
ye = std(pv2(idx_k==1));
fe = errorbar(xc,yc,ye,ye,xe,xe);
fe.LineWidth = 2;
fe.Color = 'r';
% parameters representing perfect performance
xline(p_perf(1));
yline(p_perf(2),'HandleVisibility','off');
% parameters representing first choice error
xline(p_1error(1),':');
yline(p_1error(2),':');
xlabel('Trial shift/1st correct trial');
ylabel('Learning rate/gain');
legend({'Epsilon-bias quarters','Cluster 1 (MF)','Cluster 2 (MF)','Mean/Std Cluster 1','Params: perfect','Params: 1st choice error'})
