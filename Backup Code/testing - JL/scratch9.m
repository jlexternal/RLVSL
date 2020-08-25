% scratch9.m

% Playing around with epoching and organizing pupil data
% - Jun Seok Lee

clear all
% Single subject experimental data:
isubj = 2;
load(sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)); % loads structure 'expe' to workspace


%% Goal: Define/Check for viable epoch for a given pupil dataset

% Note: The sampled times are in units of milliseconds (ms)

% pick some random pupil dataset
iblk = 4;
pupilfile = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b%02d*preproc.mat',isubj,isubj,iblk));
%rawfile = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b%02d*.mat',isubj,isubj,iblk));
%preproc_eyelink(sprintf('./Data/S%02d/%s',isubj,rawfile(1).name));
load(sprintf('./Data/S%02d/%s',isubj,pupilfile.name)); % loads structure 'data_eye' to workspace

% Note:
%       The epoch should be contained between 1) STIM1 - FBCK1 - END1   
%                                             2) STIM1 - FBCK1 - STIM2 
% Code: A check for time around the feedback points

nt = 16;
fb_inds = 3+4*[0:nt-1]; % fb index : 3 + 4k where k = [0:nt-1]
tmsg = data_eye.tmsg;
tsmp = data_eye.tsmp;
psmp = data_eye.psmp;

for it = 1:nt-1
    i_fb = fb_inds(it);
    diff1(1,it) = abs(tmsg(i_fb) - tmsg(i_fb-2)); % |FBCK1 - STIM1| 
    diff1(2,it) = abs(tmsg(i_fb) - tmsg(i_fb+1)); % |FBCK1 - END1|  
    
    diff2(1,it) = abs(tmsg(i_fb) - tmsg(i_fb-2)); % |FBCK1 - STIM1|
    diff2(2,it) = abs(tmsg(i_fb) - tmsg(i_fb+2)); % |FBCK1 - STIM2| 
end

% The epoch should be based on the lowest values coming from these values
% Note: These values change for each block so this should be registered for cropping later
min_diff1(1) = min(diff1(1,:)); min_diff1(2) = min(diff1(2,:));
min_diff2(1) = min(diff2(1,:)); min_diff2(2) = min(diff2(2,:));

%% Goal: Extract pupil data within each epoch

% Define number of samples that an epoch contains since the grain of tsmp is
% even-numbered but tmsg can be either odd or even
min_diff = [min(min_diff1(1),min_diff2(1)) min(min_diff1(2),min_diff2(2))];

t      = zeros(nt-1,1);
tbound = zeros(nt-1,2);

% Fix onset and bounds
for it = 1:nt-1 
    % if tmsg is odd, lock it to the following even number
    t(it) = tmsg(fb_inds(it));
    if mod(tmsg(fb_inds(it)),2) == 1
        t(it) = t(it)+1;
    end
    % set limits: l/u bounds = min diff bound to the next even number 
    
    tbound(it,1) = t(it) - min_diff(1); % lower limit
    if mod(tbound(it,1),2) == 1
        tbound(it,1) = tbound(it,1) + 1;
    end
    
    tbound(it,2) = t(it) + min_diff(2); % upper limit
    if mod(tbound(it,2),2) == 1
        tbound(it,2) = tbound(it,2) + 1;
    end
end

tlist = [tbound(:,1)'; t'; tbound(:,2)'];
tlist = tlist(:);
tlist_idx = zeros(length(tlist),1);

% Find indices corresponding to onset and bounds within tsmp 
% Note: To not have to search for this at every instance of a trial
i_tlist = 1;
len_tlist = length(tlist);
for ismp = 1:length(tsmp)
    if tsmp(ismp) == tlist(i_tlist)
        tlist_idx(i_tlist) = ismp;
        i_tlist = i_tlist+1;
        if i_tlist > len_tlist
            break
        end
    end
end

%% View entire pupil trajectory with markers of interest

timespan = tmsg(end)-tmsg(1);
tstart = tlist_idx(1);
tend   = tlist_idx(end);
plot(([tsmp(tstart):2:tsmp(tend)]-tsmp(tstart))/1000,psmp(tstart:tend),'LineWidth',2);
hold on;
for it = 1:nt-1
    xline((tlist(1+3*(it-1))-tsmp(tstart))/1000,'Color','g');
    xline((tlist(3*(it))-tsmp(tstart))/1000,'Color','r');
end
hold off;

%% Organize pupil data into the bins defined above
nsmp = tlist_idx(3)-tlist_idx(1)+1;
onset = tlist_idx(2) - tlist_idx(1)+1;
pupil_trace = zeros(nt-1,nsmp);

figure;
hold on;
for it = 1:nt-1
    ilb = tlist_idx(1+3*(it-1));
    iup = tlist_idx(3*(it));
    
    pupil_trace(it,:) = psmp(ilb:iup);
    
    % dumb normalization instead of drift correction/detrending
    normterm = mean(pupil_trace(it,1:onset));
    pupil_trace(it,:) = pupil_trace(it,:) - normterm;
    % FYI this isn't good.. just did it to visualise it a little better than raw
    
    
    plot((1:nsmp)-onset,pupil_trace(it,:));
    text(nsmp-100-onset,pupil_trace(it,nsmp-100)+10,num2str(expe(iblk).blck_trn(it)));
    pause(.3);
end
xline(0);
hold off;

% Plot 
figure;
shadedErrorBar([1:nsmp]-onset,mean(pupil_trace,1),std(pupil_trace,0,1));
hold on;
xline(0);


%% Goal: Analyze pupil sequence based on the condition of interest

% Code:
nblk = length(expe);

cond = 'rnd';

for iblk = 1:nblk
    if ~strcmpi(expe(iblk).type,cond)
        continue
    else
        
        
    end
end


