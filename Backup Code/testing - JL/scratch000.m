
for i = 1:10
    plot([0:.05:1],betapdf([0:.05:1],3-i*.1,4-i*.1));
    pause(.4)
end

%% datasample vs randsample testing

for n = [100 1000 10000 100000 1e6]
    w = rand(n,1);
    w = [w 1-w];
    
    fprintf('For N = %d... \n', n);
    tic;
    for i = 1:n
        a(i) = randsample([1 0], 1, true, w(i,:));
    end
    randtime = toc;
    fprintf('randsample runtime: %6f seconds\n',randtime);

    tic;
    for i = 1:n
        b(i) = datasample([1 0], 1, 'Weights', w(i,:));
    end
    datatime = toc;
    fprintf('datasample runtime: %6f seconds\n',datatime);

    fprintf('randsample %6f times faster than datasample\n\n', datatime/randtime);
end

%% Finding the standard deviation of the gaussian at mean mu for a desired False Negative Rate

fnr = .25; % desired false negative rate
func = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15)

%% Testing softmax, LogSumExp, logSoftmax
clf;
beta = 50;
x = rand(1,100);
sm = 1./(1+exp(-beta*(2*x-1)));
figure(1);
scatter(x, sm);
figure(2);
lse = log(exp(sm));

%% test
clf;
x = 1:10;
gamma = .1;
plot(x,x-(x-1)*(gamma));
hold on;
scatter(x,x+(1-gamma));
for blah = x+(1-gamma)
    xline(blah);
end















