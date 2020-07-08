% run_epiphany_qts

% Subject 17 is excluded from epiphany analysis 

prepost_epiph_rep = cell(31,2);
prepost_epiph_alt = cell(31,2);

prepost_epiph_rep{2,1}  = 1:2;      prepost_epiph_alt{2,1} = 1:2;
prepost_epiph_rep{3,1}  = [1 3];    prepost_epiph_alt{3,1} = 1:2;
prepost_epiph_rep{4,1}  = 1:2;      prepost_epiph_alt{4,1} = 1;
prepost_epiph_rep{5,1}  = 1;        prepost_epiph_alt{5,1} = 1;
prepost_epiph_rep{6,1}  = 1;        prepost_epiph_alt{6,1} = 1;
prepost_epiph_rep{7,1}  = 1;        prepost_epiph_alt{7,1} = 1;
prepost_epiph_rep{8,1}  = 1:3;      prepost_epiph_alt{8,1} = 1:3;
prepost_epiph_rep{9,1}  = 1;        prepost_epiph_alt{9,1} = 2:3;
prepost_epiph_rep{10,1} = 1;        prepost_epiph_alt{10,1} = 1;
prepost_epiph_rep{11,1} = 1:2;      prepost_epiph_alt{11,1} = 1:2;
prepost_epiph_rep{12,1} = 1;        prepost_epiph_alt{12,1} = 1:2;
prepost_epiph_rep{13,1} = 2:4;      prepost_epiph_alt{13,1} = [1 3];
prepost_epiph_rep{14,1} = 1;        prepost_epiph_alt{14,1} = 1;
prepost_epiph_rep{16,1} = 1;        prepost_epiph_alt{16,1} = 1:2;
prepost_epiph_rep{18,1} = 1:2;      prepost_epiph_alt{18,1} = [1 2 4];
prepost_epiph_rep{19,1} = 1;        prepost_epiph_alt{19,1} = 1:2;
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


