% checking found parameters for the same model on multiple fits

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);


% fitting options (fix to 'ths', 'sym', 'upd' after KFunbiased analysis)
cscheme = 'ths';
lscheme = 'sym';
nscheme = 'upd';
biastype = 'prior'; % 'epsi'        or  'prior'
biascorr = false;   % true: correct     false: subject 1st choice
selnoise = false;   % true: softmax     false: argmax
if biascorr
    bc_str = 'Corr';
else
    bc_str = 'Subj';
end
if selnoise
    sn_str = 'theta_';
else
    sn_str = '';
end
npars = 5; % kini, kinf, zeta, (epsi), (theta)
ncond = 2;
nfits = 2;
params      = squeeze(nan(npars,nsubjtot,ncond,4,nfits)); % param, subj, cond, qtr

% load and gather all the fit data
out_fit_all = squeeze(cell(ncond,4,nsubjtot,nfits));
for irun = 1:2
    for isubj = subjlist
        if irun == 1 
            filename = sprintf('./out/out_fit_KF%sbias_%sbias%s_%s%s%s_%d_%02d.mat',...
                               biastype,sn_str,bc_str,...
                               cscheme(1),lscheme(1),nscheme(1),nsubj,isubj);
        else
            filename = sprintf('./out/out_fit_KF%sbias_%sbias%s_%s%s%s_%d_%02d_2.mat',...
                               biastype,sn_str,bc_str,...
                               cscheme(1),lscheme(1),nscheme(1),nsubj,isubj);
        end
        load(filename);
        for ic = 1:2
            for iq = 1:4
                out_fit_all{ic,iq,isubj,irun} = out_fit{ic,iq,isubj};
                params(1,isubj,ic,iq,irun) = out_fit{ic,iq,isubj}.kini;
                params(2,isubj,ic,iq,irun) = out_fit{ic,iq,isubj}.kinf;
                params(3,isubj,ic,iq,irun) = out_fit{ic,iq,isubj}.zeta;
                params(4,isubj,ic,iq,irun) = out_fit{ic,iq,isubj}.epsi;
                params(5,isubj,ic,iq,irun) = out_fit{ic,iq,isubj}.theta;
            end
        end

    end
end
%% plot relation

titles = {'kini','kinf','zeta'};
figure
for ip = 1:3
    subplot(1,3,ip)
    scatter(reshape(params(ip,:,:,:,1),[],1),reshape(params(ip,:,:,:,2),[],1))
    title(titles{ip})
    if ip == 2
        xlabel('run 1','FontSize',12)
    elseif ip ==1
        ylabel('run 2','FontSize',12)
    end
end



