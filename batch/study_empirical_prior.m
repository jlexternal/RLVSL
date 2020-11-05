% study_empirical_prior


load('out_fit_noisyKF');

nsubjtot    = 31;
excluded    = [1 23 28];
subjlist    = setdiff(1:nsubjtot, excluded);
nsubj = numel(subjlist);

params_fit = nan(4,3,5,nsubj); % param, cond, time, subj
pars_trans = nan(4,3,5,nsubj);

bounds = [0 0; 0 0; 0 5; 0 10];

for isubj = 1:nsubj
    for icond = 1:2
        for itime = 1:4
            for ipar = 1:4
                params_fit(ipar,icond,itime,isubj) = out_fit_all{icond,itime,isubj}.xmap(ipar);
                if ismember(ipar,1:2)
                    pars_trans(ipar,icond,itime,isubj) = log(params_fit(ipar,icond,itime,isubj)) - log(1-params_fit(ipar,icond,itime,isubj));
                else
                    pars_trans(ipar,icond,itime,isubj) = log(params_fit(ipar,icond,itime,isubj)) - log(bounds(ipar,2)-params_fit(ipar,icond,itime,isubj));
                end
            end
        end
    end

    for ipar = 1:4
        params_fit(ipar,3,5,isubj) = out_fit_all{3,5,isubj}.xmap(ipar);
        if ismember(ipar,1:2)
            pars_trans(ipar,3,5,isubj) = log(params_fit(ipar,3,5,isubj)) - log(1-params_fit(ipar,3,5,isubj));
        else
            pars_trans(ipar,3,5,isubj) = log(params_fit(ipar,3,5,isubj)) - log(bounds(ipar,2)-params_fit(ipar,3,5,isubj));
        end
        
    end
end

%%
dist_pars = cell(1,4,3,5); % normal dist params, fit params, cond, time
parstr = {'kini','kinf','zeta','theta'};
for ipar = 1:4
    for icond = 1:2
        figure(icond)
        for itime = 1:4
            subplot(4,4,4*(itime-1)+ipar)
            histfit(squeeze(pars_trans(ipar,icond,itime,:)));
            df = fitdist(squeeze(pars_trans(ipar,icond,itime,:)),'Normal');
            dist_pars{1,ipar,icond,itime} = [df.mu,df.sigma];
            if itime == 1
                title(parstr{ipar})
            end
        end
        sgtitle(sprintf('cond: %d',icond))
    end
    
    figure(3)
    subplot(1,4,ipar)
    histfit(squeeze(pars_trans(ipar,3,5,:)));
    df = fitdist(squeeze(pars_trans(ipar,icond,itime,:)),'Normal');
    dist_pars{1,ipar,3,5} = [df.mu,df.sigma];
    title(parstr{ipar})
    sgtitle('cond: 3')
end

save('params_empirical_prior_noisyKF','dist_pars');

