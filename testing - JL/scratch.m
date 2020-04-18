%% Visualize distributions

d = 2; % distance from 50
stdev = 3; % standard deviation of distributions
nsamples = 10; 
d1 = zeros(1,nsamples);
d2 = d1;

for i = 1:nsamples
    d1(i) = normrnd(50-d, stdev);
    d2(i) = normrnd(50+d, stdev);
end
figure(10);
h1 = histogram(d1);
hold on;
xline(mean(d1),'LineWidth',2,'Color',[0 0 1]);
scatter(50-d,0,'MarkerEdgeColor',[0 0 1]);
xline(mean(d2),'LineWidth',2,'Color',[1 0 0]);
scatter(50+d,0,'MarkerEdgeColor',[1 0 0]);
xlim([0,100]);
h2 = histogram(d2);
hold off;

%% Testing perceived difficulty between the generative distributions

% Independent sampling from two Gaussians around the center with distance 2d apart
x       = [];
choice  = [];
filt    = [];
k1      = [];
k2      = [];
k1(1) = 0;
k2(1) = 0;

%initial filter variables
filt(1,1) = 50;
filt(2,1) = 50;
nu = 0;             % init process uncertainty
omega = stdev.^2;   % init observation noise
w1 = stdev.^2;       % initialized posterior variance
w2 = stdev.^2;       % initialized posterior variance

ntrials = 100;

for i = 1:ntrials
    x1 = round(normrnd(50-d,stdev));
    x2 = round(normrnd(50+d,stdev));
    x(1,i) = x1;
    x(2,i) = x2;
    
    % Choice step
    if i == 1 % 1st choice is random
        choice = randi([1 2],1);
    else 
        choice = randi([1 2],1);
        % softmax choice here
    end
    
    % Filtering step
    if i ~= 1
        if choice == 1
            k1(i) = lrate(w1, nu, omega);
            filt(1,i) = kalman(filt(1,i-1),k1(i),x1);
            filt(2,i) = filt(2,i-1);                % propagate old estimate of option 2
            w1 = (1-k1(i))*(w1+nu);
            
            k2(i) = k2(i-1);
        else
            k2(i) = lrate(w2, nu, omega);
            filt(1,i) = filt(1,i-1);                % propagate old estimate of option 1
            filt(2,i) = kalman(filt(2,i-1),k2(i),x2);
            w2 = (1-k2(i))*(w2+nu);
            
            k1(i) = k1(i-1);
        end
    end
end


figure(1);
subplot(2,1,1);
plot([1:ntrials],x(1,:),'Color',[1 0 0]); % observations bandit 1
hold on;
plot([1:ntrials],x(2,:),'Color',[0 0 1]); % observations bandit 2
scatter([1:ntrials], filt(1,:), 'MarkerEdgeColor', [.5 0 0]); % estimations bandit 1
scatter([1:ntrials], filt(2,:), 'MarkerEdgeColor', [0 0 .5]); % estimations bandit 2
yline(50-d,'Color',[.75 0 0]);
yline(50+d,'Color',[0 0 .75]);
title('Bandit and tracking dynamics');
hold off;

subplot(2,1,2);
scatter([1:ntrials],k1,'.','MarkerFaceColor',[.5 0 0],'MarkerEdgeColor', [.5 0 0]);
hold on;
scatter([1:ntrials],k2,'.','MarkerFaceColor',[0 0 .5],'MarkerEdgeColor', [0 0 .5]);
title('Learning rates');
hold off;





%%
% 2 Gaussian random walks from initial observation drawn from 2 Gaussians as above
% but with smaller initial stdev
y = [];
stdev2 = 3;
y1 = round(normrnd(50+d,stdev2));
y2 = round(normrnd(50-d,stdev2));
sig = 10;
y(1,1) = y1; y(2,1) = y2;
for i = 2:10
    y(1,i) = round(y1 + normrnd(0,sig));
    y(2,i) = round(y2 + normrnd(0,sig));
    
    k = lrate(w, nu, omega);
    if i ~= 1
        filt(1,i) = kalman(filt(1,i-1),k,y(1,i));
        filt(2,i) = kalman(filt(2,i-1),k,y(2,i));
    end
    w = (1-k)*(w+nu);
end
figure(2);
plot([1:ntrials],y(1,:),'Color',[1 0 0]); % observations bandit 1
hold on;
plot([1:ntrials],y(2,:),'Color',[0 0 1]); % observations bandit 2
scatter([1:ntrials], filt(1,:), 'MarkerEdgeColor', [.5 0 0]); % estimations bandit 1
scatter([1:ntrials], filt(2,:), 'MarkerEdgeColor', [0 0 .5]); % estimations bandit 2
plot([1:ntrials],ones(1,ntrials)*(50+d),'Color',[.75 0 0]);
plot([1:ntrials],ones(1,ntrials)*(50-d),'Color',[0 0 .75]);
hold off;

function out = kalman(x,k,o) %(previous estimate, kalman gain, observation)
    out = x+k.*(o-x);
end

function out = lrate(w,nu,omega)
    out = (w+nu)./(w+nu+omega);
end

