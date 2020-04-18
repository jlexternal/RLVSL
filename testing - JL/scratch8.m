% scratch8.m

% choices and outcomes to send to Valentin


subjlist = 2:12;
ns = numel(subjlist);
nb = 16;
nt = 16;

choices = zeros(nb,nt,ns);
outcome = zeros(nb,nt,ns);

is = 0;
for isubj = subjlist
    filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',isubj,isubj)
    if ~exist(filename,'file')
        error('Missing experiment file!');
    end
    load(filename,'expe');
    is = is + 1;
    
    ib_c = 0;
    for ib = 1:expe(1).cfg.nbout+3
        ctype = expe(ib).type;

        if ismember(ctype,'rnd')
            ib_c = ib_c + 1;

            choices(ib_c,:,is) = expe(ib).resp;
            outcome(ib_c,:,is) = expe(ib).blck_trn;
        else
            
        end
    end
end

rnd_cond = struct;
rnd_cond.outcome = outcome;
rnd_cond.choices = choices;
%save('rnd_cond','rnd_cond');
