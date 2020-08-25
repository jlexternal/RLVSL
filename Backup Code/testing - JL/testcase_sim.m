% Script for simulating the testcase experiment 
%
%
% Jun Seok Lee - Oct 2019

clear all;
clc;
% Set configuration values
d       = 3;    % distance from center for both means
stdev   = 8;    % standard deviation of the two generative distributions
ntrials = 10;   % number of trials per block
beta    = 2;    % inverse temperature parameter of softmax choice
center  = 50;   % center of the two means
n_blocks    = 10;  % number of chosen blocks desired
n_bsims      = 10000; % number of total blocks desired 

%% Generate testing blocks

% Input configuration structure for testcase_sim_blocks.m
cfg_blocks = struct;
cfg_blocks.d       = d;
cfg_blocks.stdev   = stdev;
cfg_blocks.ntrials = ntrials;
cfg_blocks.beta    = beta;
cfg_blocks.center  = center;

blocks = testcase_sim_blocks(cfg_blocks, n_blocks, n_bsims); % generate test blocks

%% Simulate decisions of optimal observer (Kalman filter)
% cases where the reward structure is "constant", "alternates", or "random"

% Set configuration values
nu      = 0;                        % process uncertainty
omega   = stdev.^2;                 % observation noise
w       = ones(2,1).*(stdev.^2);    % posterior variance(s)
beta    = 2;                        % softmax choice inverse temperature

cfg_opti        = struct;
cfg_opti.nu     = nu;
cfg_opti.omega  = omega;
cfg_opti.w      = w;
cfg_opti.beta   = beta;

% since the choices have stochasticity, need to simulate responses multiple times
% over any given block
msims = 1000;
for i = 1:size(blocks,3) % for a given block
    cfg_opti.block  = blocks(:,:,i);
    %simulate multiple times
    for j = 1:msims
        [resps(j,:,i),~,~] = testcase_sim_optiobserver(cfg_opti);
    end
    
    % get accuracy
    acc(i,:) = sum(resps(:,:,i)-1,1)/msims;
end

xc = linspace(1,0.2,n_blocks);
figure;
hold on;
for i = 1:n_blocks
    rgb = [xc(i),xc(i),1];
    plot([1:n_blocks],acc(i,:),'LineWidth',2,'Color',rgb);
end
hold off;
% calculate accuracy
%for i = 1:10 % over blocks

%end




%{
priorest    % (optional) prior learned estimates (2x1 array)
priork      %
%}






