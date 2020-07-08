function [out] = convert_fb_raw2seen(fb_raw,resp,mgen,sgen)
% Usage: Outputs the feedback seen by the subject as a consequence of his/her response 
%        via a linear transformation of the raw generated feedback of the experiment
%
% Input: fb_raw : raw generative value of the feedback
%        resp   : response of the subject
%        mgen   : mean of generative distribution of raw fb
%        sgen   : std of generative distribution of raw fb

if bsxfun(@ne,size(fb_raw),size(resp))
    error('Size of feedback structure does not match that of the responses!');
end

mnew = 55;
snew = 7.4130;

a = snew/sgen;       % slope of linear transformation aX+b
b = mnew - a*mgen;   % intercept of linear transf. aX+b

out = nan(size(fb_raw));

% loop through the array
for i = 1:length(fb_raw)
    if resp(i) ~= 1
        fb_raw(i) = -fb_raw(i); % sign the value based on choice
    end
    out(i) = round(fb_raw(i)*a+b);

    if out(i) > 99
        out = 99;
    elseif out < 1
        out = 1;
    end
end

end