function [data_eye] = preproc_eyelink(filename,cfg)
%  PREPROC_EYELINK  Preprocess Eyelink data in MAT format
%
%  Usage: [data_eye] = PREPROC_EYELINK(filename,cfg)
%
%  where filename - EyeLink data filename in MAT format
%        cfg      - configuration structure (optional)
%        data_eye - output data structure
%
%  The configuration structure can contain the following fields:
%    * fsubsmp   - sub-sampling frequency (Hz)
%    * blinkpad  - blink padding window
%    * derivpad  - derivative padding window
%    * derivthr  - instantaneous derivative threshold
%    * avgsiz    - moving average window size
%    * excthr    - exclusion threshold
%    * plotornot - plot or not?
%
%  The splinefit toolbox needs to be added to the MATLAB path before calling
%  this function. Download the toolbox at this address:
%    http://www.mathworks.com/matlabcentral/fileexchange/13812-splinefit
%
%  Valentin Wyart <valentin.wyart@ens.fr>

if nargin < 2
    cfg = [];
end
if nargin < 1
    error('Missing input filename!');
end

% check whether splinefit toolbow is in MATLAB path
if exist('splinefit') ~= 2
    error('Missing splinefit toolbox! Check help for information.');
end

% set default values for configuration parameters
if ~isfield(cfg,'fsubsmp')
    cfg.fsubsmp = [];
end
%In seconds 
if ~isfield(cfg,'blinkpad')
    cfg.blinkpad = [-0.100,+0.500];
end
if ~isfield(cfg,'derivpad')
    cfg.derivpad = [-0.200,+0.200];
end
if ~isfield(cfg,'derivthr')
    cfg.derivthr = 11;
end
if ~isfield(cfg,'avgsiz')
    cfg.avgsiz = 0.050;
end
if ~isfield(cfg,'excthr')
    cfg.excthr = 2.000;
end
if ~isfield(cfg,'plotornot')
    cfg.plotornot = true;
end

% load input filename
load(filename);

% create data structure
data_eye = [];

% get configuration parameters
fsubsmp  = cfg.fsubsmp;
%INFO: fsmp=sampling frequency
%How much you cut around a blink
blinkpad = round(cfg.blinkpad*fsmp);
%How much you cut around a too high derivative
derivpad = round(cfg.derivpad*fsmp);
derivthr = cfg.derivthr;
avgsiz   = round(cfg.avgsiz*fsmp);
excthr   = round(cfg.excthr*fsmp);

%number of samples
nsmp = length(psmp);
%Withdraw one second at the beginning and at the end
icur = fsmp+1:nsmp-fsmp;
ncur = length(icur);

%rescale the time to start at 0
tcur = tsmp(icur)-tini;
pcur = psmp(icur);
%Express derivate of the pupilla size
dcur = [0;diff(pcur)];

isbad = false(size(pcur));

% identify segments with missing pupil data
x = isnan(pcur);
%Time from which we pass from missing to nonmissing. Start of segments
d = [1;diff(x)] ~= 0;
d(isnan(d)) = false;

%Length of segments
ncon = diff([find(d);length(x)+1]);

icon = [0;cumsum(ncon(1:end-1))]+1;
%Tell if the segment is missing (xcon=1) or non missing (xcon=0)
xcon = x(d);
%Length of missing segments
ncon = ncon(xcon == 1);
%Start of missing segments
icon = icon(xcon == 1);
%Unusful
xcon = xcon(xcon == 1);

%Loop on missing segment
for i = 1:length(icon)
    %We consider that the data a little before the blink and more after the
    %blink are bad. We can parameter these lengths, but by default 0.1 sec
    %before, .5 sec after
    jbeg = max(icon(i)+blinkpad(1),1);
    jend = min(icon(i)+ncon(i)-1+blinkpad(2),ncur);
    isbad(jbeg:jend) = true;
end
pmissing = mean(isbad);
if cfg.plotornot
    fprintf('\n');
    fprintf('p(missing pupil)  = %.1f %%\n',pmissing*100);
    figure('Color','white');
end

% identify segments with quickly drifting pupil
% INFO: dcur is the derivate of pupil size
xs{1} = dcur <= 0; % shriking segments
xs{2} = dcur >= 0; % growing segments
for k = 1:2
    x = xs{k};
    d = [1;diff(x)] ~= 0;
    d(isnan(d)) = true;
    %Length of growing/shrinking segments
    ncon = diff([find(d);length(x)+1]);
    %Start of segments
    icon = [0;cumsum(ncon(1:end-1))]+1;
    %Shrinking or growing
    xcon = x(d);
    %Length of segments
    ncon = ncon(xcon == 1);
    icon = icon(xcon == 1);
    xcon = xcon(xcon == 1);
    dcon = zeros(size(ncon));
    %Compute averafe Growing/Shrinking rate
    for i = 1:length(ncon)
        dcon(i) = abs(sum(dcur(icon(i)+[0:ncon(i)-1])))/ncon(i);
    end
    if cfg.plotornot
        subplot(2,6,(k-1)*3+[1:3]);
        hold on
        xlim([0,1]);
        %Eliminate too fast drift if you parametered a maximum drifting
        %rate
        iinf = dcon < derivthr;
        %Represent in function of the time in minute the drifting rate of
        %the different segments
        plot(tcur(icon(iinf))/1000/60,dcon(iinf),'k.');
        isup = ~iinf;
        %Represent the segments which went over the maximum drifting rate
        plot(tcur(icon(isup))/1000/60,dcon(isup),'r.');
        %Represent the limit
        plot(xlim,derivthr*[1,1],'r-');
        ylim(ylim);
        hold off
        set(gca,'Layer','top','Box','off','TickDir','out');
        set(gca,'FontSize',16);
        set(gca,'XTick',0:2:10);
        set(gca,'YTick',derivthr);
        xlabel('Time (min)');
        if k == 1, ylabel('Pupil derivative'); end
        if k == 1, title('Shrinking pupil'); else title('Growing pupil'); end
    end
    %Segments where the derivative is too big
    ifilt = find(dcon > derivthr);
    for j = 1:length(ifilt)
        i = ifilt(j);    
    %We consider that the data before and after the high derivative are bad. 
    %We can parameter these lengths, but by default 0.2 sec
    %before, .2 sec after
        jbeg = max(icon(i)+derivpad(1),1);
        jend = min(icon(i)+ncon(i)-1+derivpad(2),ncur);
        isbad(jbeg:jend) = true;
    end
end
%Percetage lost due to drift
pdrifting = mean(isbad)-pmissing;
if cfg.plotornot
    fprintf('p(drifting pupil) = %04.1f %%\n',pdrifting*100);
    fprintf('--------------------------\n');
    fprintf('p(bad pupil)      = %04.1f %%\n',mean(isbad)*100);
    fprintf('\n');
end

t = tcur;
p = pcur;

isart = zeros(size(p));

% exclude bad data
p(isbad) = nan;

% smooth pupil data using moving average
%There is a moving average of a paremeterized size. By default, this is 50
%ms. The smoothing is not perfect since there is blinking and other "bad"
%segments
p = smooth(p,2*avgsiz+1,'moving');

% prepare interpolation of artifacted segments using splines
x = isbad(:);
%Time where we pass from a bad to a good segment
d = [1;diff(x)] ~= 0;
%Length of the segments
ncon = diff([find(d);length(x)+1]);
%Cumulatove count
icon = [0;cumsum(ncon(1:end-1))]+1;
%Is it a good (0) or bad (1) segment
xcon = x(d);

% mark as bad non-artifacted segments shorter than the averaging window
while true
    iout = find(xcon == 0 & ncon < 2*avgsiz,1);
    if isempty(iout)
        break
    end
    if iout == 1
        ncon(iout+1) = sum(ncon(iout:iout+1));
        ncon(iout) = [];
        icon(iout) = [];
        xcon(iout) = [];
    elseif iout == length(icon)
        ncon(iout-1) = sum(ncon(iout-1:iout));
        ncon(iout) = [];
        icon(iout) = [];
        xcon(iout) = [];
    else
        %Since the good segment is too small we can group it to the 2 bad
        %segments auround him and not consider the data
        ncon(iout-1) = sum(ncon(iout-1:iout+1));
        ncon([iout,iout+1]) = [];
        icon([iout,iout+1]) = [];
        xcon([iout,iout+1]) = [];
    end
end

% interpolate artifacted segments using splines
ncon = ncon(xcon == 1);
%take only bad segments
icon = icon(xcon == 1);
xcon = xcon(xcon == 1);

for k = 1:length(ncon)
    iart = (icon(k)-avgsiz):(icon(k)+ncon(k)+avgsiz-1);
    %ibeg = iart(1)-1;
    ibeg = iart(1);
    iend = iart(end)+1;
    if ibeg < 1 || iend > length(p)
        continue
    end
    %Check if the hole is not that big that you can't average. By default
    %it's 2 seconds time
    if iend-ibeg+1 > excthr
        p(iart) = nan;
        isart(iart) = 2;
        continue
    end
    %Construct linear constraint of the spline fit. Basically you want to set
    %the value of the spline at the beginning of the bad period and at the
    %end and also their derivative
    xc = nan(1,4);
    yc = nan(1,4);
    cc = nan(2,4);
    xc(1) = t(ibeg);
    yc(1) = p(ibeg);
    cc(:,1) = [1,0];
    xc(2) = t(ibeg);
    yc(2) = (p(ibeg)-p(ibeg-1))/(t(ibeg)-t(ibeg-1));
    cc(:,2) = [0,1];
    xc(3) = t(iend);
    yc(3) = p(iend);
    cc(:,3) = [1,0];
    xc(4) = t(iend);
    yc(4) = (p(iend+1)-p(iend))/(t(iend+1)-t(iend));
    cc(:,4) = [0,1];
    con = struct('xc',xc,'yc',yc,'cc',cc);
    %Output is a polynomial of degree 2
    pp = splinefit(t([ibeg,iend]),p([ibeg,iend]),1,con);
    %Evaluate pp in the desired values
    p(iart) = ppval(pp,t(iart));
    isart(iart) = 1;
end

if cfg.plotornot
    subplot(2,6,7:12);
    hold on
    xlim([0,50]);
    ylim([floor(min(p)/1000)*1000,ceil(max(p)/1000)*1000]);
    plot(t/1000,pcur,'k-','Color',[0.75,0.75,0.75],'LineWidth',2);
    plot(t/1000,p,'k-','LineWidth',1);
    hold off
    set(gca,'Layer','top','Box','off','TickDir','out');
    set(gca,'FontSize',16);
    xlabel('Time (s)');
    ylabel('Pupil area');
    drawnow;
end

%p is the new signal once default corrected!

% sub-sample data if requested
if ~isempty(fsubsmp)
    p = resample(p,fsubsmp,fsmp);
    t = t(1)+floor(1000/fsubsmp*[0:length(p)-1]');
    % resample also boolean arrays (a bit nasty but who cares?)
    isbad = resample(double(isbad),fsubsmp,fsmp) > 0;
    isart = resample(double(isart),fsubsmp,fsmp) > 0;
end

% fill output structure
data_eye.raw.hdr  = hdr;
data_eye.raw.fsmp = fsmp;
data_eye.raw.tini = tini;
data_eye.raw.tsmp = tsmp;
data_eye.raw.xsmp = xsmp;
data_eye.raw.ysmp = ysmp;
data_eye.raw.psmp = psmp;
data_eye.fsmp     = 500;
data_eye.tini     = tini;
data_eye.tsmp     = t+tini;
data_eye.psmp     = p;
data_eye.isbad    = isbad;
data_eye.isart    = isart;
data_eye.tsac     = tsac;
data_eye.xsac     = xsac;
data_eye.ysac     = ysac;
data_eye.tbli     = tbli;
data_eye.tmsg     = tmsg;
data_eye.smsg     = smsg;

end