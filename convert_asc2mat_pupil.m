% convert_asc2mat_pupil
% 
% This script converts subjects' eyetracked ASCII files to .MAT using function
% parse_eyelink.m
%
% Instructions: Run the script within the folder where the /Data Folder is located
%
% Jun Seok Lee
% April 2020


%% Convert ASCII files to .MAT
for isubj = 2:31
    
    eyefiles = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b*.asc',isubj,isubj));
    
    for ifile = 1:numel(eyefiles)
        filename = sprintf('./Data/S%02d/%s',isubj,eyefiles(ifile).name);
        parse_eyelink(filename);
    end
end

%% Preprocess pupil .MAT files
for isubj = 3:31
    
    eyefiles = dir(sprintf('./Data/S%02d/RLVSL_S%02d_b*.mat',isubj,isubj));
    
    for ifile = 1:numel(eyefiles)
        filename = sprintf('./Data/S%02d/%s',isubj,eyefiles(ifile).name);
        cfg = struct;
        cfg.plotornot = false;
        data_eye = preproc_eyelink(filename,cfg);
        
        savename = insertAfter(filename,length(filename)-4,'_preproc');
        save(savename,'data_eye')
    end
end

%% Rename files (maybe)


