% agglomerate_subj_data
%
% Type:         Script
%
% Objective:    To agglomerate basic subject data (seen rewards and responses) into
%               one data structure.
%
% Jun Seok Lee <jlexternal@gmail.com>

addpath('./Toolboxes');

% Load subject data
nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj       = numel(subjlist);

subj_resp_rew_all = struct;

isfirstsubj = true;
for isubj = subjlist
    jsubj = find(subjlist==isubj);
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj);
    load(filename,'expe');
    
    if isfirstsubj
        subj_resp_rew_all(1).cfg = expe(1).cfg;
        mgen = expe(1).cfg.mgen;
        sgen = expe(1).cfg.sgen;
        nt   = expe(1).cfg.ntrls;
        nb   = expe(1).cfg.nbout;
        nb_c = nb/3;
        ntrain = expe(1).cfg.ntrain;
    end
    
    resps    = nan(nb_c,nt,3);
    rews     = nan(nb_c,nt,3);
    rews_expe = nan(nb_c,nt,3);
    
    ib_c = ones(3,1);
    for ib = 1+ntrain:nb+ntrain
        ctype = expe(ib).type;
        switch ctype
            case 'rnd' % random across successive blocks
                ic = 3;
            case 'alt' % always alternating
                ic = 2;
            case 'rep' % always the same
                ic = 1;
        end
        resp = expe(ib).resp;
        resps(ib_c(ic),:,ic) = resp;
        
        rew  = expe(ib).blck;
        rews(ib_c(ic),:,ic) = convert_fb_raw2seen(rew,resp,mgen,sgen)/100;
        rews_expe(ib_c(ic),:,ic) = expe(ib).blck_trn;
        
        ib_c(ic) = ib_c(ic)+1;
    end
    
    subj_resp_rew_all(isubj).resp     = resps;
    subj_resp_rew_all(isubj).rew_seen = rews;
    subj_resp_rew_all(isubj).rew_expe  = rews_expe;
end

save('subj_resp_rew_all','subj_resp_rew_all');