%%
clear all java
close all hidden
clc

addpath ./Toolboxes/IO
addpath ./Toolboxes/Stimuli/Visual


% set screen parameters
iscr = 1; % screen index
res  = []; % screen resolution
fps  = []; % screen refresh rate
ppd  = 40; % number of screen pixels per degree of visual angle
syncflip = true; % change to true/false

% % % % set list of color-wise R/G/B values
% % % color_rgb = [ ...
% % %     255,255,000; ... % yellow
% % %     255,130,000; ... % orange
% % %     255,000,000; ... % red
% % %     170,000,170; ... % purple
% % %     000,000,255; ... % blue
% % %     035,220,000; ... % green
% % %     ]/255;
% % % color_opa = 2/3; % color opacity

% set stimulation parameters
lumibg    = 128/255; % background luminance
% % % fixtn_siz = 0.3*ppd; % fixation point size
% % % shape_siz = 4.0*ppd; % shape size
% % % shape_off = 4.0*ppd; % shape offset
% % % fbtxt_fac = 2.5; % feedback text magnification factor
% % % intxt_fac = 1.5; % instruction text magnification factor


% create video structure
video = [];

try
    
    % hide cursor and stop spilling key presses into MATLAB windows
    HideCursor;
    FlushEvents;
    ListenChar(2);
    
    % check keyboard responsiveness before doing anything
    fprintf('\n');
    fprintf('Press any key to check keyboard responsiveness... ');
    if WaitKeyPress([],10) == 0
        fprintf('\n\n');
        error('No key press detected after 10 seconds.');
    else
        fprintf('Good.\n\n');
    end
    
    % set keys
    KbName('UnifyKeyNames');
    keywait = KbName('space');
    keyquit = KbName('ESCAPE');
    keyresp = KbName({'E','P'});
    
    % open main window
    % set screen resolution and refresh rate
    if ~isempty(res) && ~isempty(fps)
        r = Screen('Resolutions',iscr);
        i = find([r.width] == res(1) & [r.height] == res(2));
        if isempty(i) || ~any([r(i).hz] == fps)
            error('Cannot set screen to %d x %d at %d Hz.',res(1),res(2),fps);
        end
        Screen('Resolution',iscr,res(1),res(2),fps);
    end
    % set screen synchronization properties
    % see 'help SyncTrouble',
    %     'help BeampositionQueries' or
    %     'help ConserveVRAMSettings' for more information
    if syncflip
        if ispc
            % soften synchronization test requirements
            Screen('Preference','SyncTestSettings',[],[],0.2,10);
            % enforce beamposition workaround for missing VBL interval
            Screen('Preference','ConserveVRAM',bitor(4096,Screen('Preference','ConserveVRAM')));
        end
        Screen('Preference','VisualDebuglevel',3);
    else
        % skip synchronization tests altogether
        Screen('Preference','SkipSyncTests',1);
        Screen('Preference','VisualDebuglevel',0);
        Screen('Preference','SuppressAllWarnings',1);
    end
    % set font properties
    if ismac
        txtfnt = 'Helvetica';
        txtsiz = round(1.0*ppd);
    elseif ispc
        txtfnt = 'Arial'; % closest to Helvetica
        txtsiz = round(2/3*ppd); % text size is ~2/3 smaller in Windows than MacOSX
    end
    Screen('Preference','TextAlphaBlending',1);
    Screen('Preference','DefaultFontName',txtfnt);
    Screen('Preference','DefaultFontSize',txtsiz);
    Screen('Preference','DefaultFontStyle',0);
    % prepare configuration and open main window
    PsychImaging('PrepareConfiguration');
    PsychImaging('AddTask','General','UseFastOffscreenWindows');
    PsychImaging('AddTask','General','NormalizedHighresColorRange');
    video.i = iscr;
    video.res = Screen('Resolution',video.i);
    video.h = PsychImaging('OpenWindow',video.i,0);
    [video.x,video.y] = Screen('WindowSize',video.h);
    video.ifi = Screen('GetFlipInterval',video.h,100,50e-6,10);
    Screen('BlendFunction',video.h,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    Priority(MaxPriority(video.h));
    Screen('ColorRange',video.h,1);
    Screen('FillRect',video.h,lumibg);
    Screen('Flip',video.h);
    
    % check screen refresh rate
    if ~isempty(fps) && fps > 0 && round(1/video.ifi) ~= fps
        error('Screen refresh rate not equal to expected %d Hz.',fps);
    end
    
    
    
    
    % close Psychtoolbox
    Priority(0);
    Screen('CloseAll');
    FlushEvents;
    ListenChar(0);
    ShowCursor;
    
catch
    
    % close Psychtoolbox
    Priority(0);
    Screen('CloseAll');
    FlushEvents;
    ListenChar(0);
    ShowCursor;
    
    % handle error
    rethrow(lasterror);
    
end