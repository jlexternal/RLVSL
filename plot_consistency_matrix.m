% Choice consistency matrix
clear all;
close all;

nsubjtot    = 31;
excluded    = [1];
subjlist    = setdiff(1:nsubjtot, excluded);
subparsubjs = [excluded 15 20 23 28];
subjlist    = setdiff(1:nsubjtot, subparsubjs); % if excluding underperforming/people who didn't get it

ns = numel(subjlist);
nb = 16;
nt = 16;
nc = 3;

choices = zeros(nb,nt,nc,ns);

is = 0;
for isubj = subjlist
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)
    if ~exist(filename,'file')
        error('Missing experiment file!');
    end
    load(filename,'expe');
    is = is + 1;
    
    ib_c = zeros(3,1);
    
    for ib = 4:expe(1).cfg.nbout+3
        ctype = expe(ib).type;
        switch ctype
            case 'rnd' % random across successive blocks
                ic = 3;
            case 'alt' % always alternating
                ic = 2;
            case 'rep' % always the same
                ic = 1;
        end
        ib_c(ic) = ib_c(ic) + 1;
        choices(ib_c(ic),:,ic,is) = expe(ib).resp; 
    end
end

% choice consistency matrices
nt   = 16;  % number of trials in block/sequence
nq   = 4;   % number of quarters... this is obvious but don't want to hard code
nb_q = 4;   % number of blocks per quarter

consist_mat         = zeros(nt,nt,nc,nq); % (t_1, t_2, conditions, quarters)
similar_mat_blck    = nan(nt,nt,nq);
similar_mat_subj    = nan(nt,nt,ns);
similar_mat         = nan(nt,nt,nq,nc);

for ic = 1:3 % loop through conditions
    for iq = 1:4 % loop through quarters
        for is = 1:ns % loop through subjects
            resps(1:4,:) = -(choices((4*iq)-3:(4*iq),:,ic,is)-2);   % store quarter data (converting 2's to 0)
            resps_z      = complex(resps, double(~resps));          % convert to complex vectors
            
            for ib_q = 1:4 % loop through blocks within quarter
                similar_mat_blck(:,:,ib_q) = real(resps_z(ib_q,:)'.*resps_z(ib_q,:)); % take real value after mat. multiplication
            end
            similar_mat_subj(:,:,is) = sum(similar_mat_blck,3)/nb_q; % average similarity per number of blocks/sequences
        end
        similar_mat(:,:,iq,ic) = sum(similar_mat_subj,3)/ns; % average similarity across subjects
    end
end

ctype = {'rep','alt','rnd'};
iplot = 1;
colormap('hot');
for ic = 1:3
    for iq = 1:4
        if ismember(iplot,[5,10,15])
            iplot = iplot+1;
        end
        subplot(nc,nq+1,iplot);
        x = similar_mat(:,:,iq,ic);
        imagesc(x);
        imagesc(triu(x,1));
        if ic == 1
            title(sprintf('Quarter %d',iq));
        end
        if iq == 1
            ylabel(sprintf('Condition: %s',ctype{ic}));
        end
        iplot = iplot + 1;
    end
end
subplot(3,5,[5 10 15]);
colorbar;
sgtitle('Choice consistency curves across quarters and conditions (avg''d across subjects)');





