% scratch10.m

% Playing around with detrending pupil data
% - Jun Seok Lee

clear all

% Add toolbox paths
addpath('./Toolboxes/NoiseTools/');

% Single subject experimental data:
isubj = 2;
load(sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)); % loads structure 'expe' to workspace
nt = expe(1).cfg.ntrls;
% pick some random pupil dataset
ib = 4;
pupilfile = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b%02d*preproc.mat',isubj,isubj,ib));
load(sprintf('./Data/S%02d/%s',isubj,pupilfile.name)); % loads structure 'data_eye' to workspace

%% Identify epochs 
idx_fb = 3+4*[0:nt-1]; % fb index : 3 + 4k where k = [0:nt-1]
tmsg = data_eye.tmsg;
smsg = data_eye.smsg;
tsmp = data_eye.tsmp;
psmp = data_eye.psmp;

for it = 1:nt-1
    i_fb = idx_fb(it);
    diff1(1,it) = abs(tmsg(i_fb) - tmsg(i_fb-2)); % |FBCK1 - STIM1| 
    diff1(2,it) = abs(tmsg(i_fb) - tmsg(i_fb+1)); % |FBCK1 - END1|  
    
    diff2(1,it) = abs(tmsg(i_fb) - tmsg(i_fb-2)); % |FBCK1 - STIM1|
    diff2(2,it) = abs(tmsg(i_fb) - tmsg(i_fb+2)); % |FBCK1 - STIM2| 
end

% The epoch should be based on the lowest values coming from these values
% Note: These values change for each block so this should be registered for cropping later
min_diff1(1) = min(diff1(1,:)); min_diff1(2) = min(diff1(2,:));
min_diff2(1) = min(diff2(1,:)); min_diff2(2) = min(diff2(2,:));
min_diff = [min(min_diff1(1),min_diff2(1)) min(min_diff1(2),min_diff2(2))];
clearvars min_diff1 min_diff2

%% plot the raw data and preproc'd

timespan_raw = data_eye.raw.tsmp;
t_init       = timespan_raw(1);
timespan_ppc = data_eye.tsmp;
timespan_ppc = (timespan_ppc-t_init)/1000;
timespan_raw = (timespan_raw-t_init)/1000;

pupil_raw = data_eye.raw.psmp;
pupil_ppc = data_eye.psmp;
close all;
figure(1);
subplot(2,1,1);
hold on;
plot(timespan_raw,pupil_raw,'LineWidth',1);
plot(timespan_ppc,pupil_ppc,'LineWidth',2);

%% test different detrending curve fits

% Identify nan in data
idx_nan = isnan(timespan_ppc);
idx_nan = idx_nan | isnan(pupil_ppc);

% %% Dumb Method 1: Fit polynomial of degree nt-1 %%
%                   Logic behind nt-1 is that there are nt trials, so nt-1 inflection points
fit_poly = polyfit(timespan_ppc(~idx_nan),pupil_ppc(~idx_nan),nt-1);
figure(1);
plot(timespan_ppc(~idx_nan),polyval(fit_poly,timespan_ppc(~idx_nan)),'--','Color','k');
hold on;
title('Raw and pre-processed pupil data');
legend('raw','preproc''d','polynomial fit','Location','southeast');

% %% Smart Method: Robust detrending of de Cheveigné 
%
[fit_robust,w,r] = nt_detrend(pupil_ppc(~idx_nan),nt-1);

% need to sync this data with the full time sample set (nan included) since it does
% not consider it. 
pupil_detrend_rbst = nan(length(idx_nan),1);
ptr_dtd  = 1;
for is = 1:length(idx_nan)
    if idx_nan(is) == 1
        
    else
        pupil_detrend_rbst(is) = fit_robust(ptr_dtd);
        ptr_dtd = ptr_dtd + 1;
    end
    
end



%% Plot the detrended data
pupil_detrend_poly = pupil_ppc(~idx_nan)-polyval(fit_poly,timespan_ppc(~idx_nan));
ymax = max(max(pupil_detrend_poly),max(fit_robust))+50;

figure(1);
subplot(2,1,2);
% plot epochs
for i_fb = idx_fb
    t_fb = tmsg(i_fb)-t_init;
    ep_start = (t_fb-min_diff(1))/1000;
    ep_end   = (t_fb+min_diff(2))/1000;
    patch([ep_start ep_start ep_end ep_end],[-ymax ymax ymax -ymax],[.9 .9 .9],'EdgeColor','none','FaceAlpha',.7,'HandleVisibility','off');
end
yline(0,'HandleVisibility','off');
hold on;
plot(timespan_ppc(~idx_nan),pupil_detrend_poly,'LineWidth',2);
plot(timespan_ppc,pupil_detrend_rbst,'LineWidth',1,'Color','b');
legend('polyfit','robustfit');
ylim([-ymax ymax]);
title('Detrended pupil data (polynomial order nt-1); epochs shaded');
hold off;

%% Group trends based on feedback value

% Locate indices of the fb timepoint and save traces in separate structure
pupil_trace = zeros(nt,min_diff(2)+min_diff(1));
idx_fb_tsmp = zeros(nt,1);
it = 1;
for i_tsmp = 1:length(tsmp)-1
    if tmsg(idx_fb(it)) >= tsmp(i_tsmp) && tmsg(idx_fb(it)) < tsmp(i_tsmp+1)
        idx_fb_tsmp(it) = i_tsmp;
        pupil_trace(it,:) = pupil_detrend_rbst(idx_fb_tsmp(it)-min_diff(1):idx_fb_tsmp(it)+min_diff(2)-1);
        if it == nt
            break
        end
        it = it + 1;
    end
end


fb = expe(ib).blck_trn;
idx_fb_pos = fb >= 50;

figure(2);
shadedErrorBar(1:size(pupil_trace,2),mean(pupil_trace(idx_fb_pos,:),1),std(pupil_trace(idx_fb_pos,:)),...
                'lineprops',{'Color',[.8 .2 .2],'LineWidth',2},'patchSaturation',0.075);
hold on;
shadedErrorBar(1:size(pupil_trace,2),mean(pupil_trace(~idx_fb_pos,:),1),std(pupil_trace(~idx_fb_pos,:)),...
                'lineprops',{'Color',[.2 .2 .8],'LineWidth',2},'patchSaturation',0.075);
hold off;




