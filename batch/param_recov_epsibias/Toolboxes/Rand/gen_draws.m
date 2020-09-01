function [out] = gen_draws(davg,dlim,nepi)
% GEN_DRAWS  Generate random draws
%
% Usage:
%   [out] = gen_draws(davg,dlim,nepi)
%
% Valentin Wyart <valentin.wyart@ens.fr>

nper = 5e3; % number of random permutations
plotornot = false; % plot results or not?

dmin = dlim(1); % minimum distance between reversals
dmax = dlim(2); % maximum distance between reversals

if davg < get_davg(1-eps) || davg > get_davg(eps)
    error('requested distance between reversals out of bounds!');
end

% get p(reversal)
prev = fzero(@(p)get_davg(p)-davg,0.5);

% plot p(reversal) wrt distance between reversals
if plotornot
    pvec = 0.01:0.01:0.50;
    dvec = nan(size(pvec));
    for i = 1:length(pvec)
        dvec(i) = get_davg(pvec(i));
    end
    figure('Color','white');
    hold on
    xlim([0,0.5]);
    plot(pvec,dvec,'k-','LineWidth',2);
    ylim(ylim);
    plot(prev*[1,1],ylim,'r-');
    plot(prev,davg,'ko','MarkerSize',12,'MarkerFaceColor',[1,0.5,0.5]);
    hold off
    set(gca,'Layer','top','Box','off','TickDir','out');
    set(gca,'FontName','Helvetica','FontSize',16);
    set(gca,'XTick',0:0.1:0.5);
    set(gca,'YTick',0:2:20);
    xlabel('p(reversal)');
    ylabel('distance between reversals');
end

% get theoretical pdf/cdf
ps = prev*(1-prev).^(1:dmax);
ps(1:dmin-1) = 0;
ps = ps/sum(ps);
cs = cumsum(ps);

% sample distances between reversals
xs = nan(nper,nepi);
pr = nan(nper,1);
for iper = 1:nper
    % sample distances between reversals
    for iepi = 1:nepi
        xs(iper,iepi) = find(rand < cs,1,'first');
    end
    % fit p(reversal) wrt sampled distances between reversals
    fm = @(p,x)-sum(log(p)+log(1-p)*x)+nepi*log(sum(p*(1-p).^(dmin:dmax)));
    pr(iper) = fminbnd(@(p)fm(p,xs(iper,:)),0.001,0.999);
end
ns = sum(xs,2);

% sort wrt offset between desired and observed p(reversal)
[~,is] = sort(abs(log(pr)-log(prev)));
xs_all = xs(is,:);
ns_all = ns(is);
pr_all = pr(is);

% filter sequences with desired number of sequences
is = 1:find(ns_all ~= nepi*davg,1)-1;
xs_all = xs_all(is,:);
ns_all = ns_all(is);
pr_all = pr_all(is);

nseq = size(xs_all,1);

found = false;
for iseq = 1:nseq
    xs = xs_all(iseq,:);
    ns = ns_all(iseq);
    pr = pr_all(iseq);
    for iper = 1:nper
        xs = xs(randperm(nepi));
        ys = unpack(xs);
        if ...
                abs(diff(hist(ys(1:ns/2),[1,2]))) < dmin && ...
                abs(diff(hist(ys(ns/2+1:end),[1,2]))) < dmin
            if mod(nepi,2) > 0
                found = true;
                break
            elseif abs(sum(xs(1:nepi/2))-ns/2) < dmin
                found = true;
                break
            end
        end
    end
    if found
        break
    end
end
if ~found
    error('could not generate draws!');
end

% get biased p(reversal)
fm = @(p,x)-sum(log(p)+log(1-p)*x);
pb = fminbnd(@(p)fm(p,xs),0.001,0.999);

out      = [];
out.davg = davg;
out.dlim = dlim;
out.nepi = nepi;
out.prev = prev;
out.xs   = xs;
out.pr   = pr;
out.pb   = pb;
out.ys   = ys;

if plotornot
    % show sequence using imagesc
    % with pink and blue colors
end

    function [d] = get_davg(p)
    ps = p*(1-p).^(dmin:dmax);
    ps = ps/sum(ps);
    d = sum(ps.*(dmin:dmax));
    end

    function [y] = unpack(x)
    n = sum(x);
    y = ones(1,n);
    for i = 1:length(x)-1
        j = sum(x(1:i))+1;
        y(j:end) = 3-y(j);
    end
    end

end