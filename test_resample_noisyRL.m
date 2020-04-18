
%% Show how the resampling method for the noisy RL particle filter works
%  We have a fast method that only requires to sample from a truncated univariate
%  normal distribution. This would not work for more than two alternatives.
clear all
close all
clc

n = 1e5; % number of samples

% pick means and standard deviations randomly
m = normrnd(zeros(1,2),ones(1,2))
s = gamrnd(ones(1,2),ones(1,2))

% sample from truncated bivariate normal distribution
% this is slow (because there is no fast method for sampling from this distribution)
x = mvnrnd(repmat(m,[n,1]),[s(1)^2,0;0,s(2)^2]);    % this approximates the joint distribution
i = x(:,1)-x(:,2) > 0; % truncate wrt X1-X2         % this truncates the distribution where x2<x1
x = x(i,:);                                         

% version that only requires to sample from truncated univariate normal distribution
% this is fast (because there exists fast methods for sampling from this distribution)
% 1/ sample X1-X2 from truncated univariate normal distribution
y = normrnd(repmat(m(1)-m(2),[n,1]),sqrt(s(1)^2+s(2)^2));
y_nt = y;
i = y > 0; % truncate
y = y(i);   % sampled values of (x1-x2)
% 2/ resample X1
ALPHA   = s(1)^2 / (s(1)^2 + s(2)^2);               % 
MU      = m(1)   - ALPHA*(m(1)-m(2));               % mean of new x1 distribution
SIGMA   = sqrt(s(1)^2 - ALPHA^2 * (s(1)^2+s(2)^2)); % s.d. of new x1 distribution
y1 = ALPHA*y+normrnd(repmat(MU,[numel(y),1]),SIGMA);
% 3/ reconstruct X2 as X1 - (X1-X2)
y2 = y1-y;
% get the reconstructed samples
y = [y1,y2];

%% plot results of the two methods
figure;
subplot(1,2,1);
histogram(x(:,1),'Normalization','pdf','DisplayStyle','stairs');
hold on
histogram(y(:,1),'Normalization','pdf','DisplayStyle','stairs');
hold off
subplot(1,2,2);
histogram(x(:,2),'Normalization','pdf','DisplayStyle','stairs');
hold on
histogram(y(:,2),'Normalization','pdf','DisplayStyle','stairs');
%histogram(y2b,'Normalization','pdf','DisplayStyle','stairs');
hold off
