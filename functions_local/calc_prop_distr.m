function out = calc_prop_distr(cfg)
% calc_prop_distr
%
% Function: Calculates the distribution curves for the three measures
%               1/ p(correct)
%               2/ p(repeat previous choice)
%               3/ p(repeat 1st choice)
%
% Input: cfg - configuration structure with 
%               1/ response matrix      (numerical matrix)
%               2/ type of proportion   (string)
%
% Output: out - output structure with
%               1/ p    : (mean) proportion
%               (if ns > 1)
%               2/ s    : spread around mean proportion
%               3/ p_i  : individual proportions of each sim sample (for bootstrapping in fitter)
%
% Jun Seok Lee <jlexternal@gmail.com>

if ~isfield(cfg,'resp')
    error('Response matrix missing!');
end
if ~ismember(cfg.type,{'correct','repprev','repfirst'})
    error('Proportion type not recognized!');
end

resp = cfg.resp;
type = cfg.type;

resp(resp==2) = 0;

nb = size(resp,1);
nt = size(resp,2);
ns = size(resp,3);

% Determine whether base curve or simulation curve (w/ spread)
if numel(size(resp)) > 2
    hasspread = true;
else
    hasspread = false;
end

out = struct;
switch type
    case 'correct'
        % calculates proportion correct; nt values
        if hasspread
            p_i = mean(resp,1); % individual proportions for each sim
            p   = mean(p_i,3);
            s   = std(p_i,1,3)+eps;
        else
            p = mean(resp,1);
        end
        
    case 'repprev'
        % calculates proportion of repeat choices to previous trial; nt-1 values
        prevrespeq = bsxfun(@eq,resp(:,2:end,:),resp(:,1:end-1,:));
        if hasspread
            p_i = mean(prevrespeq,1);
            p   = mean(p_i,3);
            s   = std(p_i,1,3)+eps;
        else
            p = mean(prevrespeq,1);
        end
        
    case 'repfirst'
        % calculates proportion of repeat choices to 1st trial; nt-1 values
        firstresp = resp;
        firstresp(:,2:end,:) = firstresp(:,1,:).*ones(nb,nt-1,ns);
        firstrespeq = bsxfun(@eq,resp(:,2:end,:),firstresp(:,2:end,:));
        if hasspread
            p_i = mean(firstrespeq,1);
            p   = mean(p_i,3);
            s   = std(p_i,1,3)+eps;
        else
            p = mean(firstrespeq,1);
        end
end

% output individual sim proportions, mean proportion (and spread)
if hasspread
    out.p_i = p_i;
    out.p = p;
    out.s = s;
else
    out.p = p;
end

end