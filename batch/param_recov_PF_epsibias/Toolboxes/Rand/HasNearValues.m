function [noccurences] = HasNearValues(x,nnear,xexcluded)

if nargin < 3
    xexcluded = [];
end
if nargin < 2
    nnear = 2;
end
if nargin < 1
    error('Wrong input argument list.');
end

xval = unique(x);

if ~isempty(xexcluded)
    xval = setdiff(xval,xexcluded);
end

noccurences = 0;
for i = 1:length(xval)
    noccurences = noccurences+nnz(diff(find(x == xval(i))) <= nnear);
end

end