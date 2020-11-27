% run_epiphany_qts

% Subject 15, 17, and 20 are excluded from epiphany analysis 

% Assumptions:
%   1/ A post-epiphany quarter is one where the subject is more than 60% confident
%   2/ The 1st quarter is always considered pre-epiphany 
%   3/ A subject can revert their epiphany if confidence decreases after an increase

prepost_epiph_rep = cell(31,2);
prepost_epiph_alt = cell(31,2);

prepost_epiph_rep{2,1}  = 1;        prepost_epiph_alt{2,1} = 1:2;
prepost_epiph_rep{3,1}  = [1 3];    prepost_epiph_alt{3,1} = 1:2;
prepost_epiph_rep{4,1}  = 1:2;      prepost_epiph_alt{4,1} = 1;
prepost_epiph_rep{5,1}  = 1;        prepost_epiph_alt{5,1} = 1;
prepost_epiph_rep{6,1}  = 1;        prepost_epiph_alt{6,1} = 1;
prepost_epiph_rep{7,1}  = 1;        prepost_epiph_alt{7,1} = 1;
prepost_epiph_rep{8,1}  = 1:3;      prepost_epiph_alt{8,1} = 1:3;
prepost_epiph_rep{9,1}  = 1;        prepost_epiph_alt{9,1} = [1 4];
prepost_epiph_rep{10,1} = 1;        prepost_epiph_alt{10,1} = 1;
prepost_epiph_rep{11,1} = 1:2;      prepost_epiph_alt{11,1} = 1:2;
prepost_epiph_rep{12,1} = 1;        prepost_epiph_alt{12,1} = 1:2;
prepost_epiph_rep{13,1} = 1;        prepost_epiph_alt{13,1} = [1 3];
prepost_epiph_rep{14,1} = 1;        prepost_epiph_alt{14,1} = 1;
prepost_epiph_rep{16,1} = 1;        prepost_epiph_alt{16,1} = 1:2;
prepost_epiph_rep{18,1} = 1:2;      prepost_epiph_alt{18,1} = [1 2 4];
prepost_epiph_rep{19,1} = 1;        prepost_epiph_alt{19,1} = 1;
prepost_epiph_rep{21,1} = 1:2;      prepost_epiph_alt{21,1} = 1:2;
prepost_epiph_rep{22,1} = 1;        prepost_epiph_alt{22,1} = 1;
prepost_epiph_rep{24,1} = 1;        prepost_epiph_alt{24,1} = 1;
prepost_epiph_rep{25,1} = 1;        prepost_epiph_alt{25,1} = 1:2;
prepost_epiph_rep{26,1} = 1;        prepost_epiph_alt{26,1} = 1;
prepost_epiph_rep{27,1} = [1 3];    prepost_epiph_alt{27,1} = 1;
prepost_epiph_rep{29,1} = 1;        prepost_epiph_alt{29,1} = 1;
prepost_epiph_rep{30,1} = 1;        prepost_epiph_alt{30,1} = 1:2;
prepost_epiph_rep{31,1} = 1;        prepost_epiph_alt{31,1} = 1:3;

% fill in the gaps
for isubj = 1:31
    if isempty(prepost_epiph_rep{isubj,1})
        continue;
    end
    prepost_epiph_rep{isubj,2} = setdiff(1:4,prepost_epiph_rep{isubj,1}(:)');
    prepost_epiph_alt{isubj,2} = setdiff(1:4,prepost_epiph_alt{isubj,1}(:)');
    
end


conf = nan(2,4,nsubj); % rep/alt, quarter, subjs

conf(:,:,2) = [1 1 1 1;     0 .6 1 1];
conf(:,:,3) = [.8 .9 .6 .8; .5 .7 .9 .9];
conf(:,:,4) = [.6 .7 1 1;   0 .9 1 1];
conf(:,:,5) = [.8 .95 1 1;   .8 .95 1 1];
conf(:,:,6) = [.65 .8 .9 .9;.4 .8 .9 .99];
conf(:,:,7) = [1 1 1 1;     .9 .8 .8 1];
conf(:,:,8) = [.5 .3 .5 1;  .5 .2 .2 1];
conf(:,:,9) = [0 .8 .9 1;   0 .7 .6 .3];
conf(:,:,10)= [.2 1 1 1;    .4 1 1 1];
conf(:,:,11)= [.3 .3 .9 .9; .7 .6 .9 .9];
conf(:,:,12)= [0 .8 1 1;    0 .7 .8 .9];
conf(:,:,13)= [0 .9 .8 1;   0 .8 .6 1;];
conf(:,:,14)= [.9 .95 .95 1;.7 .8 .75 .9];
conf(:,:,16)= [0 1 1 1;     0 .65 .8 1];
conf(:,:,18)= [0 0 1 1;     0 .5 .8 0];
conf(:,:,19)= [.5 1 1 1;    .5 .6 1 1];
conf(:,:,21)= [0 .75 .9 .99;0 .5 .9 .99];
conf(:,:,22)= [.9 1 1 1;    .5 1 1 1];
conf(:,:,24)= [0 .8 .95 1;  0 .8 .93 1];
conf(:,:,25)= [.6 .9 1 1;   0 .7 1 1];
conf(:,:,26)= [0 1 1 1;     0 1 1 1];
conf(:,:,27)= [0 .8 .5 1;   0 1 .8 1];
conf(:,:,29)= [1 1 1 1;     1 1 1 1];
conf(:,:,30)= [.6 .9 1 1;   .3 .7 .8 1];
conf(:,:,31)= [.8 .9 .95 .95;.4 .2 .2 .6];












