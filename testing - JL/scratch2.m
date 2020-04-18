% This scratch code has separate filters for each option, tracking them separately
% independent of one another

%% Visualize distributions

d = 3; % distance from 50
stdev = 4; % standard deviation of distributions
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

% Declare variables 
x       = [];
choice  = [];
estis   = [];
k       = [];

% Initial filter and estimation variables
estis(1,1) = 50;        % init estimation of choice 1 at 50 (flat prior)
estis(2,1) = 50;        % init estimation of choice 2 at 50 (flat prior)
k(1,1) = 0;
k(2,1) = 0;
nu      = 0;            % init process uncertainty
omega   = stdev.^2;     % init observation noise
w1      = stdev.^2;     % initialized posterior variance for choice estimate 1
w2      = stdev.^2;     % initialized posterior variance for choice estimate 2

% Softmax parameters
beta = 3;

ntrials = 10;
for i = 1:ntrials
    % Choice step (softmax)
    prob1       = beta.*estis(1,i);
    prob2       = beta.*estis(2,i);
    weights     = [exp(prob1) exp(prob2)]./exp(prob1+prob2);
    choice(i)   = datasample([1 2], 1, 'Replace', true, 'Weights', weights);
    
    % Outcome sampling step
    if choice(i) == 1
        x(1,i) = round(normrnd(50-d,stdev));
        x(2,i) = NaN;
    else
        x(1,i) = NaN;
        x(2,i) = round(normrnd(50+d,stdev));
    end
    
    % Estimation step
    if choice(i) == 1
        estis(1,i+1)    = kalman(estis(1,i),k(1,i),x(1,i));
        estis(2,i+1)    = estis(2,i);                % propagate old estimate of option 2
    else
        estis(1,i+1)    = estis(1,i);                % propagate old estimate of option 1
        estis(2,i+1)    = kalman(estis(2,i),k(2,i),x(2,i));
    end
    
    % Filtering step
    if choice(i) == 1
        k(1,i+1)        = lrate(w1, nu, omega);
        w1              = (1-k(1,i+1)).*(w1+nu);
        k(2,i+1)        = k(2,i);
    else
        k(2,i+1)        = lrate(w2, nu, omega);
        w2              = (1-k(2,i+1)).*(w2+nu);
        k(1,i+1)        = k(1,i);
    end

end

%should replace non-update tracking values w/ NaN
choicepts = choice;
choicepts(choicepts==1) = 50-d;
choicepts(choicepts==2) = 50+d;

%plots

figure(1);
subplot(2,1,1);
scatter([1:ntrials],x(1,:),'MarkerEdgeColor',[1 0 0]); % observations bandit 1
hold on;
scatter([1:ntrials],x(2,:),'MarkerEdgeColor',[0 0 1]); % observations bandit 2
plot([1:ntrials+1]-1, estis(1,:), 'Color', [.5 0 0]); % estimations bandit 1
plot([1:ntrials+1]-1, estis(2,:), 'Color', [0 0 .5]); % estimations bandit 2
scatter([1:ntrials]-1, choicepts,'x','MarkerEdgeColor',[0 0 0],'LineWidth',2);
yline(50-d,'--','Color',[.75 0 0]);
yline(50+d,'--','Color',[0 0 .75]);
ylim([40 60]);
title('Bandit, tracking, choice dynamics');
hold off;

subplot(2,1,2);
scatter([1:ntrials+1]-1,k(1,:),'o','MarkerFaceColor',[.5 0 0],'MarkerEdgeColor', [.5 0 0]);
hold on;
scatter([1:ntrials+1]-1,k(2,:),'o','MarkerFaceColor',[0 0 .5],'MarkerEdgeColor', [0 0 .5]);
title('Learning rates');
hold off;

% Local functions
function out = kalman(x,k,o) %(previous estimate, kalman gain, observation)
    out = x+k.*(o-x);
end

function out = lrate(w,nu,omega)
    out = (w+nu)./(w+nu+omega);
end

