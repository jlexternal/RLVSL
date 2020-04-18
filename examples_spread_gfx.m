% pretty spread graphing

% 1/ Generate some random data

ny = 10;
nx = 20;
xdata = [1:nx]';
ys = xdata*ones(1,ny)+4*rand([nx ny]); % generate random data
ydata = mean(ys,2);
yerr  = std(ys,0,2);

ys2 = xdata*ones(1,ny)*.8+4*rand([nx ny]); % generate random data
ydata2 = mean(ys2,2);
yerr2  = std(ys2,0,2);

ys3 = xdata*ones(1,ny)*1.4+4*rand([nx ny]); % generate random data
ydata3 = mean(ys3,2);
yerr3  = std(ys3,0,2);
% 2/ Plot

ifig = 1;

figure(ifig);
ifig = ifig + 1;
hold on;
errorbar(xdata,ydata,yerr); % regular errorbar
shadedErrorBar(xdata,ydata,yerr,'lineprops','g'); % shaded errorbar
shadedErrorBar(xdata,ydata2,yerr2,'lineprops',{'Color',[.43 .5 .2]}); % custom color
shadedErrorBar(xdata,ydata3,yerr3,'lineprops',{'Color',[.3 .4 .6]},'patchSaturation',.1); % changing transparency level
hold off;
