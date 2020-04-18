%% Generate a test file

subj_i = 97; % subject number

gen_expe_subj_rlvsl(subj_i); % generates and saves to file the structure

%% 

clear all java 
close all hidden
clc   
  
feature('DefaultCharacterSet', 'UTF8'); % for MAc-PC compatibilty when sharing code between machines

% add toolboxes 
addpath ./Toolboxes/Rand/
addpath ./Toolboxes/Stimuli/Visual/
addpath ./Toolboxes/IO/  

% get participant information    
argindlg = inputdlg({'Subject number'},'RLVSL',1,{'','','','',''});
if isempty(argindlg)
    error('experiment cancelled!');
end
subj = str2num(argindlg{1}); 

% run experiment
[expe,aborted] = run_expe_rlvsl(subj,false,false);  % use during testing/debug
%expe,aborted] = run_expe_rlvsl(subj,true,true);    % use for real exp
