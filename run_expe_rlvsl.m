function [expe,aborted,errmsg] = run_expe_rlvsl(subj,syncflip,trackeye)

% Advisory: Before running the experiment on actual participants, be sure to go
%           through the code searching for 'debug' in any commented sections. These must either be
%           re-enabled or disabled for proper functionality and data collection - J Lee.

% toggle fastmode       to go through the experiment faster but still at a reasonable pace
% toogle superfastmode  to go through the experiment at lightning speed
fastmode        = true; % debug
superfastmode   = false; % debug
eyecalib        = false; % here for debugging purposes

% check input arguments
if nargin < 3
    trackeye = false;
end
if nargin < 2
    syncflip = true;
end
if nargin < 1
    error('Missing subject number!');
end

if trackeye && (fastmode || superfastmode)
    error('Turn off fastmode!');
end

if fastmode || superfastmode
    beep on;
    for i = 1:5
        disp('Fast mode or Superfast mode is turned on! Abandon experiment if testing real participants!!');
        beep;
        pause(.1);
    end
    fastpausemult   = 0.25; % makes the pauses much shorter
else
    fastpausemult   = 1; 
end

% create header
hdr = [];
hdr.subj = subj;
hdr.date = datestr(now,'yyyymmdd-HHMM');

% load experiment structure
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',subj,subj)
if ~exist(filename,'file')
    error('Missing experiment file!');
end
load(filename,'expe');

% define output arguments
aborted = false;    % aborted prematurely?
errmsg = [];        % error message

% determine sex of participant
argigend = inputdlg({'Gender of participant (0:male, 1:female)'},'RLVSL',1,{'','','','',''}); % using 0 and 1 to avoid keyboard confusion
if ~ismember(argigend,{'0','1'})
    error('Unexpected input. Enter either 0 or 1.');
end

subj_gndr = str2num(argigend{1}); % screen index (change to 0 if no extended screens)
if subj_gndr == 0
    subj_gndr = 'm';
else
    subj_gndr = 'f';
end
hdr.gender = subj_gndr;

expe(1).hdr = hdr;

% get and set screen parameters
argiscr = inputdlg({'Select screen (0:main, 1:extended)'},'RLVSL',1,{'','','','',''});
if ~ismember(argiscr,{'0','1'})
    error('Unexpected input. Enter either 0 or 1.');
end
iscr = str2num(argiscr{1}); % screen index (change to 0 if no extended screens)
res  = []; % screen resolution
fps  = []; % screen refresh rate
ppd  = 40; % number of screen pixels per degree of visual angle
ppd_mult = 1.5; % for use in the instructions loading functions

% set list of color-wise R/G/B values
color_rgb   = [ ...  % inner rectangle
    251,229,214; ... % red
    222,235,247; ... % blue
    222,235,247; ... % grey
    222,235,247; ... % grey
    ]/255;
lever_rgb = [ ...
    251,229,214; ... % red
    222,235,247; ... % blue
    192,192,192; ... % grey
    192,192,192; ... % grey
    ]/255;
color_frame = [ ...  % darker outside border rectangle
    244,177,131; ...
    157,195,230; ...
    100,100,100; ...
    100,100,100; ...
    ]/255;

% set stimulation parameters
lumibg    = 128/255;    % background luminance
fixtn_siz = 0.3*ppd;    % fixation point size
shape_siz = 4.0*ppd;    % shape size
tools_siz = 2.0*ppd;    % tools and gears size
shape_off = 4.0*ppd;    % shape offset
fbtxt_fac = 1.5;        % feedback text magnification factor
intxt_fac = 1.5;        % instruction text magnification factor
prog_off    = ppd*8.0;  % bonus texture offset

% trial-to-trial RL difficulty calculation
fnr      = .25; % Desired false negative rate of higher distribution
func     = @(sig)fnr-normcdf(50,55,sig);
sig_opti = fzero(func,15);

% create video structure
video = [];

try
    % hide cursor and prevent spilling key presses into MATLAB windows
    HideCursor;  % debug point (enable when finished debugging)
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
    keyquit = KbName('X'); %JL: change abort key since i don't have an ESC
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
    
    % screen synchronization properties
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
        txtsiz = round(0.75*ppd);
    elseif ispc
        txtfnt = 'Arial';           % closest to Helvetica
        txtsiz = round(1.0*ppd);    % text size is ~2/3 smaller in Windows than MacOSX
    end
    
    % PsychToolbox properties
    Screen('Preference','TextEncodingLocale','UTF-8');
    Screen('Preference','TextAlphaBlending',1);
    Screen('Preference','TextRenderer', 1);
    Screen('Preference','DefaultFontName',txtfnt);
    Screen('Preference','DefaultFontSize',txtsiz);
    Screen('Preference','DefaultFontStyle',0)
    %Screen('Preference', 'SkipSyncTests', 1);  %%% debug point - disabled for testing - JL 
    % prepare configuration and open main window
    PsychImaging('PrepareConfiguration');
    PsychImaging('AddTask','General','UseFastOffscreenWindows');
    PsychImaging('AddTask','General','NormalizedHighresColorRange');
    video.i     = iscr;
    video.res   = Screen('Resolution',video.i);
    video.h     = PsychImaging('OpenWindow',video.i,0); 
    [video.x,video.y]   = Screen('WindowSize',video.h);
    video.ifi           = Screen('GetFlipInterval',video.h,100,50e-6,10);
    Screen('BlendFunction',video.h,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    Priority(MaxPriority(video.h));
    Screen('ColorRange',video.h,1);
    Screen('FillRect',video.h,lumibg);
    Screen('Flip',video.h);
    
    % check screen refresh rate
    if ~isempty(fps) && fps > 0 && round(1/video.ifi) ~= fps
        error('Screen refresh rate not equal to expected %d Hz.',fps);
    end
    
    % open offscreen window
    video.hoff = Screen('OpenOffscreenWindow',video.h);
    
    % initialize eye-tracker connection
    if trackeye
        if EyelinkInit() ~= 1
            error('could not initialize eye-tracker connection!');
        end
        [~,ev] = Eyelink('GetTrackerVersion');
        fprintf('Connection to %s eye-tracker initialized successfully.\n',ev);
        el = EyelinkInitDefaults(video.h);
        % disable sounds during eye-tracker calibration to avoid nasty conflicts!
        el.targetbeep = 0;
        el.feedbackbeep = 0;
        EyelinkUpdateDefaults(el);
        % update camera setup
        Eyelink('Command','active_eye = LEFT');                 % LEFT or RIGHT
        Eyelink('Command','binocular_enabled = NO');            % YES:binocular or NO:monocular
        Eyelink('Command','simulation_screen_distance = 560');  % in mm
        Eyelink('Command','use_ellipse_fitter = NO');           % YES:ellipse or NO:centroid
        Eyelink('Command','pupil_size_diameter = YES');         % YES:diameter or NO:area
        Eyelink('Command','sample_rate = 500');                 % 1000 or 500 or 250
        Eyelink('Command','elcl_tt_power = 2');                 % 1:100% or 2:75% or 3:50%
        % update calibration parameters
        Eyelink('Command','calibration_type = HV5');            % HV5 or HV9
        Eyelink('Command','generate_default_targets = NO');     % YES:default or NO:custom
        cnt = [video.x/2,video.y/2];
        off = ppd*8;
        pnt = zeros(5,2);
        pnt(1,:) = cnt;
        pnt(2,:) = cnt-[0,off];
        pnt(3,:) = cnt+[0,off];
        pnt(4,:) = cnt-[off,0];
        pnt(5,:) = cnt+[off,0];
        pnt = num2cell(reshape(pnt',[],1));
        Eyelink('Command','calibration_samples = 6');
        Eyelink('Command','calibration_sequence = 0,1,2,3,4,5');
        Eyelink('Command','calibration_targets = %d,%d %d,%d %d,%d %d,%d %d,%d',pnt{:});
        Eyelink('Command','validation_samples = 5');
        Eyelink('Command','validation_sequence = 0,1,2,3,4,5');
        Eyelink('Command','validation_targets = %d,%d %d,%d %d,%d %d,%d %d,%d',pnt{:});
        % update file output parameters
        Eyelink('Command','file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE');
        Eyelink('Command','file_sample_data = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS');
    end
    
    %%%%%%%%%%%%%%%%% BEGIN texture loading and positioning code %%%%%%%%%%%%%%%%%%%
    % shape loading code 
    if true 
    % Note: The PNG files must be in GREYSCALE format
    
    % Load bandit shape textures
    shape_tex = zeros(2,12); 
    trnfb_tex = zeros(2,6);
    for is = 1:12
        img     = double(imread(sprintf('./img/shape%d.png',is)))/255;  % unchosen stimuli
        img     = imresize(img,shape_siz/size(img,1));
        shape_tex(1,is) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
        img     = double(imread(sprintf('./img/shape%dc.png',is)))/255; % chosen stimuli
        img     = imresize(img,shape_siz/size(img,1));
        shape_tex(2,is) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
        
        % Load training letter shape textures for overlay on feedback
        if is > 6
            img     = double(imread(sprintf('./img/shape%d.png',is)))/255;  % unchosen stimuli
            img     = imresize(img,(1.5*ppd)/size(img,1));
            trnfb_tex(1,is-6) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
        end
    end
    
    % Load bandit shape textures for the 'rnd' condition
    %   Note: these are divided into "shapes" and "symbols" since some are more
    %   block-y and others are more script-y
    shape_tex_rnd1 = zeros(2,16);
    shape_tex_rnd2 = zeros(2,16);
    for is = 1:16
        % "shapes"
        img     = double(imread(sprintf('./img/shape_rnd%d.png',is)))/255;  % unchosen stimuli
        img     = imresize(img,shape_siz/size(img,1));
        shape_tex_rnd1(1,is) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
        img     = double(imread(sprintf('./img/shape_rnd%dc.png',is)))/255; % chosen stimuli
        img     = imresize(img,shape_siz/size(img,1));
        shape_tex_rnd1(2,is) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
        % "symbols"
        img     = double(imread(sprintf('./img/symbol_rnd%d.png',is)))/255;  % unchosen stimuli
        img     = imresize(img,shape_siz/size(img,1));
        shape_tex_rnd2(1,is) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
        img     = double(imread(sprintf('./img/symbol_rnd%dc.png',is)))/255; % chosen stimuli
        img     = imresize(img,shape_siz/size(img,1));
        shape_tex_rnd2(2,is) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    rnd_shape_order = randperm(16); % randomize order of the random condition shape textures
    irnd = 1;
    
    % Position bandit shape textures
    shape_rec       = zeros(2,4); % shape_rec(1,:)=left shape_rec(2,:)=right
    shape_rec(1,:)  = CenterRectOnPoint(Screen('Rect',shape_tex(1,1)),video.x/2-shape_off,video.y/2); % left
    shape_rec(2,:)  = CenterRectOnPoint(Screen('Rect',shape_tex(1,1)),video.x/2+shape_off,video.y/2); % right
    
    % Create background rectangles and outer frames
    bg_rect             = [video.x/2-(1.5*shape_off) video.y/2-shape_off/2 video.x/2+(1.5*shape_off) video.y/2+shape_off/2];
    bg_rect_frame       = [video.x/2-(1.55*shape_off) video.y/2-shape_off/1.8 video.x/2+(1.55*shape_off) video.y/2+shape_off/1.8];
    bg_rect_cntrd       = CenterRectOnPoint(bg_rect,video.x/2,video.y/2);
    bg_rect_frame_cntrd = CenterRectOnPoint(bg_rect_frame,video.x/2,video.y/2);
    
    % Load technician shape textures
    techn_combis = [1 1; 1 2; 1 3; ...  % all permutations of m/f technician combinations
                    2 1; 2 2; 2 3; ...
                    3 1; 3 2; 3 3];   
    ig = mod(subj,9)+1; % pseudorandomizing choice of technicians
    % set color-technician pairings
    techn_order = techn_combis(ig,:);
    expe(1).techn_order = techn_order;
    techn_tex = zeros(2,2); 
    img     = double(imread(sprintf('./img/techn%dm.png',techn_combis(ig,1))))/255; 
    img     = imresize(img,shape_siz/size(img,1));
    techn_tex(1,1) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    img     = double(imread(sprintf('./img/techn%df.png',techn_combis(ig,2))))/255; 
    img     = imresize(img,shape_siz/size(img,1));
    techn_tex(2,1) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    
    % Position technician shape textures
    techn_rec = zeros(1,4);
    techn_rec = CenterRectOnPoint(Screen('Rect',techn_tex(1,1)),video.x/2,video.y/2-shape_off/1.8-shape_siz/2);
    
    % Load tools/gears shape textures
    calib_tex = zeros(2,1); % calib_tex(1)=tools, calib_tex(2)=gears
    for it = 1:2
        img     = double(imread(sprintf('./img/calib%d.png',it)))/255;
        img     = imresize(img,tools_siz/size(img,1));
        calib_tex(it,1) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    
    % Position tools/gears shape textures
    calib_rec       = zeros(2,4); % calib_rec(1,:)=left, calib_rec(2,:)=right
    calib_rec(1,:)  = CenterRectOnPoint(Screen('Rect',calib_tex(1,1)),video.x/2-shape_off/1.3,video.y/2-shape_off/1.8-shape_siz/1.3); % left
    calib_rec(2,:)  = CenterRectOnPoint(Screen('Rect',calib_tex(1,1)),video.x/2+shape_off/1.3,video.y/2-shape_off/1.8-shape_siz/1.3); % right
    
    
    % Load lever textures
    lever_tex = zeros(4,1); % 1-left,up; 2-left,down; 3-right,up; 4-right,down;
    for is = 1:4
        img = double(imread(sprintf('./img/lever%d.png',is)))/255; 
        img = imresize(img,shape_siz/size(img,1));
        lever_tex(is,1) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    
    % Position lever textures
    lever_rec = zeros(4,4); % 1-left,up; 2-left,down; 3-right,up; 4-right,down;
    lever_rec(1,:) = CenterRectOnPoint(Screen('Rect',lever_tex(1,1)),video.x/2-(1.55*shape_off)-87,video.y/2-shape_siz/2);
    lever_rec(2,:) = CenterRectOnPoint(Screen('Rect',lever_tex(2,1)),video.x/2-(1.55*shape_off)-87,video.y/2+shape_siz/2);
    lever_rec(3,:) = CenterRectOnPoint(Screen('Rect',lever_tex(3,1)),video.x/2+(1.55*shape_off)+86,video.y/2-shape_siz/2);
    lever_rec(4,:) = CenterRectOnPoint(Screen('Rect',lever_tex(4,1)),video.x/2+(1.55*shape_off)+86,video.y/2+shape_siz/2);
    end
    %%%%%%%% END texture loading and positioning code %%%%%%%%
    
    % create fixation point
    img         = CreateCircularAperture(fixtn_siz);
    fixtn_tex   = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    fixtn_rec   = CenterRectOnPoint(Screen('Rect',fixtn_tex(1)),video.x/2,video.y/2);
    
    % identify number of practice and real blocks
    nblk        = length(expe);
    isprac      = cellfun(@(s)strcmp(s,'training'),{expe.type});
    nblk_prac   = nnz(isprac);    % number of practice blocks
    nblk_expe   = nnz(~isprac);   % number of true testing blocks
    
    acc_score = [];
    
    for iblk = 1:nblk %block loop
        
        % import block structure values
        blck = [];
        blck.blck  = expe(iblk).blck;  % sampled values (unaltered)
        blck.shape = expe(iblk).shape; % blck.shape(1)- correct shape, blck.shape(2)- incorrect shape
        blck.color = expe(iblk).color; % background color corresponding to condition
        blck.locat = expe(iblk).locat; % corresponds to location of stimulus
        blck.bcond = expe(iblk).type;  % block condition
        if strcmpi(blck.bcond,'rnd')
            blck.color = 4; % force the random color to grey that was reserved for random condition
        end
        
        ntrl = length(expe(iblk).blck);
        
        % initialize response structure
        resp        = zeros(1,ntrl);    % response (1:correct, 2:incorrect)
        rt          = zeros(1,ntrl);    % response time (seconds)
        pupilmissed = zeros(1,ntrl);    % (0: pupil there on trial, 1: pupil was missing at beginning of trial)
        
        % load 'rnd' condition shape textures
        if strcmpi(blck.bcond,'rnd')
            if mod(irnd,2)==1
                shape_tex(1,blck.shape(1)) = shape_tex_rnd1(1,rnd_shape_order(irnd));
                shape_tex(1,blck.shape(2)) = shape_tex_rnd1(1,rnd_shape_order(irnd+1));
                shape_tex(2,blck.shape(1)) = shape_tex_rnd1(2,rnd_shape_order(irnd));
                shape_tex(2,blck.shape(2)) = shape_tex_rnd1(2,rnd_shape_order(irnd+1));
            else
                shape_tex(1,blck.shape(1)) = shape_tex_rnd2(1,rnd_shape_order(irnd-1));
                shape_tex(1,blck.shape(2)) = shape_tex_rnd2(1,rnd_shape_order(irnd));
                shape_tex(2,blck.shape(1)) = shape_tex_rnd2(2,rnd_shape_order(irnd-1));
                shape_tex(2,blck.shape(2)) = shape_tex_rnd2(2,rnd_shape_order(irnd));
            end
            irnd = irnd+1;
        end
        % draw start screen
        Screen('TextStyle',video.h,0);
        Screen('TextSize',video.h,txtsiz);
        
        if isprac(iblk)
            if iblk == 1
                if trackeye
                    % calibrate eye-tracker
                    EyelinkDoTrackerSetup(el,el.ENTER_KEY);
                    t = Screen('Flip',video.h);
                end
                
                % load instructions screen
                load_instructions_pre1;
                Screen('DrawingFinished',video.h);
                Screen('Flip',video.h);
                pause(2*fastpausemult);
                WaitKeyPress(keywait,[],false); % press space to move on
                
                load_instructions_pre2;
                Screen('DrawingFinished',video.h);
                Screen('Flip',video.h);
                pause(2*fastpausemult);
                WaitKeyPress(keywait,[],false); % press space to move on
                
                % load pre-practice screen
                load_pre_prac;
                Screen('DrawingFinished',video.h);
                Screen('Flip',video.h);
                pause(2*fastpausemult);
                WaitKeyPress(keywait,[],false); % press space to move on
            end
            
            pointsarr = zeros(1,ntrl); % to display at the end of training blocks
            
            disp(['For proctor purposes: ' num2str(blck.shape)]);
        else % real blocks
            if iblk == nblk_prac + 1
                labeltxt = 'Appuyez sur [espace] pour commencer le jeu.';
                labelrec = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt),video.x/2,round(1.2*ppd));
                Screen('DrawText',video.h,labeltxt,labelrec(1),labelrec(2),0);
                Screen('TextSize',video.h,round(txtsiz*intxt_fac));
                Screen('Flip',video.h);
                pause(2*fastpausemult);
                WaitKeyPress(keywait,[],false); % press space to move on to next block
                if trackeye
                    % calibrate eye-tracker
                    EyelinkDoTrackerSetup(el,el.ENTER_KEY);
                    t = Screen('Flip',video.h);
                end
            end
            
            % display technician calibrating machine
            if iblk < nblk
                ic_calib = blck.color;
                for icalib = 1:6
                    % block appropriate technician avatar
                    if ~strcmpi(blck.bcond,'rnd')
                        if strcmpi(blck.bcond,'rep')
                            itechn = 1;
                        elseif strcmpi(blck.bcond,'alt')
                            itechn = 2;
                        end
                        draw_techn(itechn); % normal technician avatar % 1 or 2 as input
                    end
                    Screen('FillRect', video.h, color_frame(ic_calib,:), bg_rect_frame_cntrd);
                    Screen('FillRect', video.h, color_rgb(ic_calib,:), bg_rect_cntrd);
                    lever_up = true;
                    draw_levers(lever_rgb(ic_calib,:));
                    draw_calib(icalib);
                    Screen('DrawingFinished',video.h);
                    Screen('Flip',video.h);
                    pause(.3*fastpausemult);
                end
                pause(.2);
                if ~strcmpi(blck.bcond,'rnd')
                    draw_techn(itechn); % normal technician avatar
                end
                Screen('FillRect', video.h, color_frame(ic_calib,:), bg_rect_frame_cntrd);
                Screen('FillRect', video.h, color_rgb(ic_calib,:), bg_rect_cntrd);
                draw_calib(icalib);
                lever_up = true;
                draw_levers(lever_rgb(ic_calib,:));
                labeltxt = 'Appuyez sur [espace] pour jouer.';
                labelrec = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt),video.x/2,20.0*ppd);
                Screen('DrawText',video.h,labeltxt,labelrec(1),labelrec(2),0);
                Screen('DrawingFinished',video.h);
                pause(2*fastpausemult);
                Screen('Flip',video.h);
                WaitKeyPress(keywait,[],false); % press space to move on to next block
            end
        end
        
        % draw fixation point
        Screen('FillRect', video.h, color_frame(blck.color,:), bg_rect_frame_cntrd);
        Screen('FillRect', video.h, color_rgb(blck.color,:), bg_rect_cntrd);
        lever_up = true;
        draw_levers(lever_rgb(blck.color,:));
        Screen('DrawTexture',video.h,fixtn_tex,[],fixtn_rec,[],[],[],0);
        Screen('DrawingFinished',video.h);
        
        if trackeye
            % start recording
            Eyelink('OpenFile','RLVSL');
            Eyelink('StartRecording');
            eye_used = Eyelink('EyeAvailable');  % get eye that's tracked
            % wait for 5 extra seconds
            t = Screen('Flip',video.h,t+roundfp(5.000));
        else 
            t = Screen('Flip',video.h);
        end
        
        for itrl = 1:ntrl
            
            if CheckKeyPress(keyquit) % check for abort
                aborted = true;
                break;
            end
            
            lever_up = true; % status of lever positions (up)
            % place correct shape in its randomly chosen position
            if blck.locat(itrl) == 1 % if correct place is on the left
                il = 1;              % left position gets correct shape
                ir = 2;
            else        % if correct place is on the right
                il = 2; % right position gets correct shape
                ir = 1;
            end
            
            % check for good positioning of pupil and log if missing
            if trackeye
                missedpupil = check_pupil(trackeye,eye_used);
                if missedpupil
                    pupilmissed(itrl) = 1;
                end
            end
            
            draw_stim;
            Screen('DrawTexture',video.h,fixtn_tex,[],fixtn_rec,[],[],[],0);
            draw_levers(lever_rgb(blck.color,:));
            if isprac(iblk) % show key associations during training blocks
                key_training;
            end
            
            if fastmode || superfastmode
                t = Screen('Flip',video.h,t+roundfp(0.000,0.000));  % trigger only when testing
            else
                t = Screen('Flip',video.h,t+roundfp(2.000,0.500));  % for the real experimental situation
            end
            if trackeye
                eyemess = sprintf('TRL%02d_STIM',itrl);
                Eyelink('Message',eyemess);
            end
            
            % get participant response (1:correct, 2:incorrect)
            [key,tkey] = WaitKeyPress(keyresp,[],false);
            if key == 1
                resp(itrl) = il;
            else
                resp(itrl) = ir;
            end
            rt(itrl) = tkey-t;                                  % timer end
            lever_up = false;   % status of lever position (down)
            if trackeye
                eyemess = sprintf('TRL%02d_RESP',itrl);
                Eyelink('Message',eyemess);
            end
            
            % calculate feedback (value)
            ease_mult = 1.5;    % how much easier the training blocks should be in terms of mean difference multiple 
            mu_new   = 55;  % desired mean of correct shape distribution
            if isprac(iblk)   % remapping of value difference for practice blocks
                mu_new  = 50+(abs(mu_new-50)*ease_mult);   % desired mean of distribution
            end
            sig_new = sig_opti;   % desired std of distribution
            a = sig_new/expe(1).cfg.sgen;       % slope of linear transformation aX+b
            b = mu_new - a*expe(1).cfg.mgen;    % intercept of linear transf. aX+b
            
            fb_val = blck.blck(itrl);
            
            % FIX: might want to put the transformed value rather than untransformed
            blck.blck(itrl) = round(fb_val*a+b); % unsigned transformed value to store in expe structure
            if resp(itrl) ~= 1
                fb_val = fb_val*-1; % sign the value based on choice
            end
            % rescale random variable distr. from N(.1,.2^2) to N(55,sig^2)
            fb_val = round(fb_val*a+b);
            % clip extreme values
            if fb_val > 99
                fb_val = 99;
            elseif fb_val < 1
                fb_val = 1;
            end
            
            % log outcome values for training 
            if isprac(iblk)
                pointsarr(itrl) = fb_val;
            end
            
            % show feedback
            draw_stim;
            draw_levers(lever_rgb(blck.color,:));
            draw_feedback;
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h);
            if trackeye
                eyemess = sprintf('TRL%02d_FBCK',itrl);
                Eyelink('Message',eyemess);
            end
            if superfastmode
                continue;
            elseif fastmode
                pause(.5);
            else
                pause(1);  % lag point
            end
            %clear the shapes
            Screen('FillRect', video.h, color_frame(ic,:), bg_rect_frame_cntrd);
            Screen('FillRect', video.h, color_rgb(ic,:), bg_rect_cntrd);
            lever_up = true;
            draw_levers(lever_rgb(blck.color,:));
            Screen('Flip',video.h);
            if trackeye
                eyemess = sprintf('TRL%02d_END',itrl);
                Eyelink('Message',eyemess);
            end
            
            
        end % end of trial loop
        
        acc_score = [acc_score; resp]; % store accuracy scores for each block in array
            
        % update expe struct.
        expe(iblk).resp         = resp;
        expe(iblk).rt           = rt;
        expe(iblk).pupilmissed  = pupilmissed;
        expe(iblk).blck_trn     = blck.blck;    % trn = "transformed"
        
        if trackeye
            % wait for 5 extra seconds
            draw_stim;
            Screen('FillRect', video.h, color_frame(blck.color,:), bg_rect_frame_cntrd);
            Screen('FillRect', video.h, color_rgb(blck.color,:), bg_rect_cntrd);
            lever_up = true;
            draw_levers(lever_rgb(blck.color,:));
            Screen('DrawTexture',video.h,fixtn_tex,[],fixtn_rec,[],[],[],0);
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h,t+roundfp(5.000));
            % stop recording
            Eyelink('StopRecording');
            Eyelink('CloseFile');
            % retrieve eye-tracker datafile
            if ~aborted
                fpath = sprintf('./Data/S%02d',hdr.subj);
                fname = sprintf('RLVSL_S%02d_b%02d_%s',hdr.subj,iblk,datestr(now,'yyyymmdd-HHMM'));
                fname = fullfile(fpath,fname);
                for i = 1:10
                    status = Eyelink('ReceiveFile',[],[fname,'.edf']);
                    if status >= 0 % should be >
                        break
                    end
                end
                if status < 0 % should be <=
                    warning('could not retrieve eye-tracker datafile %s!',fname);
                else
                    save([fname,'.mat'],'expe');
                end
            end
            % redraw default textures
            Screen('FillRect', video.h, color_frame(blck.color,:), bg_rect_frame_cntrd);
            Screen('FillRect', video.h, color_rgb(blck.color,:), bg_rect_cntrd);
            lever_up = true;
            draw_levers(lever_rgb(blck.color,:));
            Screen('Flip',video.h);
        end
        
        if aborted
            break;
        end
        
        % end of training blocks
        if iblk < nblk_prac+1
            draw_feedback_trn(resp,pointsarr);
            labeltxt1 = 'La meilleure lettre dans cette manche était :';
            trn_shape_rec = CenterRectOnPoint(Screen('Rect',shape_tex(1,1)),video.x/2,6.0*ppd); % center good letter
            labeltxt2 = sprintf('Vous l’avez choisie à %.1f%% des tirages.', numel(resp(resp==1))/ntrl*100); % accuracy for that block
            labeltxt3 = '(Les carrés blancs correspondent aux bons choix)';
            labeltxt4 = 'Appuyez sur [espace] pour continuer.';
            
            Screen('TextSize',video.h,round(txtsiz));
            labelrec3 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt3),video.x/2,video.y/2+1.5*ppd);
            Screen('TextSize',video.h,round(txtsiz*fbtxt_fac));
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,3.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2);
            labelrec4 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt4),video.x/2,20.0*ppd);
            
            Screen('TextSize',video.h,round(txtsiz));
            Screen('DrawText',video.h,labeltxt3,labelrec3(1),labelrec3(2),0);
            Screen('TextSize',video.h,round(txtsiz*fbtxt_fac));
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawTexture',video.h,shape_tex(1,blck.shape(1)),[],trn_shape_rec,[],[],[],0); % draw good letter up top and center
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),0);
            Screen('DrawText',video.h,labeltxt4,labelrec4(1),labelrec4(2),0);
            
            if iblk == nblk_prac
                Screen('DrawingFinished',video.h);
                Screen('Flip',video.h);
                pause(1*fastpausemult);
                WaitKeyPress(keywait,[],false); % press space to move on
            end
        end
        
        % end of block screen
        if iblk == nblk_prac % end of training session screen
            labeltxt1 = 'Fin d''entraînement.';
            labeltxt2 = 'Appuyez sur [espace] pour continuer aux sessions vraies.';
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,video.y/2-3.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2+3.0*ppd);
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),0);
            acc = calc_acc;
            draw_acc;                       % acc screen
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h);
            pause(1*fastpausemult);
            WaitKeyPress(keywait,[],false);
            load_instructions_post1;        % inst1
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h);
            pause(2*fastpausemult);
            WaitKeyPress(keywait,[],false); 
            load_instructions_post2;        % inst2
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h);
            pause(2*fastpausemult);
            WaitKeyPress(keywait,[],false); 
            load_instructions_post3;        % inst3

        elseif iblk == length(expe) % end of last session
            labeltxt1 = 'Fin d''expérience.';
            labeltxt2 = 'Merci d''avoir participé.';
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,video.y/2-3.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2+3.0*ppd);
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),0);
            acc = calc_acc;
            draw_acc;
        elseif mod(expe(iblk).sesh,2) == 1 && mod(iblk-nblk_prac,(length(expe)-nblk_prac)/expe(length(expe)).sesh)==0 % end of a session but not break time (odd)
            labeltxt1 = sprintf('Fin de séance %d.',expe(iblk).sesh);
            labeltxt2 = 'Faites une petite pause pour reposer vos yeux et votre cou.';
            labeltxt3 = 'Appuyez sur [espace] pour continuer.';
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,video.y/2-3.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2-1.0*ppd);
            labelrec3 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt3),video.x/2,video.y/2+3.0*ppd);
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),0);
            Screen('DrawText',video.h,labeltxt3,labelrec3(1),labelrec3(2),0);
            acc = calc_acc;
            draw_acc;
        elseif mod(expe(iblk).sesh,2) == 0 && mod(iblk-nblk_prac,(length(expe)-nblk_prac)/expe(length(expe)).sesh)==0 % break time screen (even)
            labeltxt1 = sprintf('Fin de séance %d.',expe(iblk).sesh);
            labeltxt2 = 'Ne continuez pas.';
            labeltxt3 = 'Appelez le expérimentateur.';
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,video.y/2-4.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2);
            labelrec3 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt3),video.x/2,video.y/2+4.0*ppd);
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),1);
            Screen('DrawText',video.h,labeltxt3,labelrec3(1),labelrec3(2),0);
            acc = calc_acc;
            draw_acc;
            eyecalib = true;

        elseif iblk > nblk_prac % not end of session
            labeltxt = 'Appuyez sur [espace] pour continuer.';
            labelrec = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt),video.x/2,round(1.2*ppd));
            Screen('DrawText',video.h,labeltxt,labelrec(1),labelrec(2),0);
        end
        Screen('TextSize',video.h,round(txtsiz*intxt_fac));
        Screen('DrawingFinished',video.h);
        Screen('Flip',video.h);
        pause(1*fastpausemult);
        WaitKeyPress(keywait,[],false); % press space to move on to next block
            
        if eyecalib && trackeye  
            % calibrate eye-tracker
            EyelinkDoTrackerSetup(el,el.ENTER_KEY);
            eyecalib = false;
            t = Screen('Flip',video.h);
        end
        
        
    end % end of block loop
    
    if trackeye
        % reset to default calibration targets
        Eyelink('Command','generate_default_targets = YES');
        % close eye-tracker connection
        Eyelink('Shutdown');
    end
    
    % End of task - display goodbye message, close Psychtoolbox
    Priority(0);
    Screen('CloseAll');
    FlushEvents;
    ListenChar(0);
    ShowCursor;
    
catch
    if trackeye && exist('el','var')
        % stop recording
        Eyelink('StopRecording');
        % reset to default calibration targets
        Eyelink('Command','generate_default_targets = YES');
        % close eye-tracker connection
        Eyelink('Shutdown');
    end
    
    % close Psychtoolbox
    Priority(0);
    Screen('CloseAll');
    FlushEvents;
    ListenChar(0);
    ShowCursor;
    
    % handle error
    if nargout > 2
        errmsg = lasterror;
        errmsg = rmfield(errmsg,'stack');
    else
        rethrow(lasterror);
    end
end

function [t] = roundfp(t,dt)
    % apply duration rounding policy for video flips
    % where t  - desired (input)/rounded (output) duration
    %       dt - desired uniform jitter on duration (default: none)
    n = round(t/video.ifi);
    % apply uniform jitter
    if nargin > 1 && dt > 0
        m = round(dt/video.ifi);
        n = n+ceil((m*2+1)*rand)-(m+1);
    end
    % convert frames to duration
    t = (n-0.5)*video.ifi;
end
    
function [x] = bit2dec(b)
    % convert binary array to decimal
    x = sum(b(:).*2.^(0:numel(b)-1)');
end

function acc_out = calc_acc
    % calculate accuracy
    acc_score(acc_score == 2) = 0; % all wrong answers to 0 
    acc_score_disp = sum(sum(acc_score));
    acc_out = acc_score_disp/(numel(acc_score));
    acc_score = []; % reset accuracy score array
end

function draw_acc
    acc_txt = sprintf('%.1f%% correct',acc*100);
    acc_rec = CenterRectOnPoint(Screen('TextBounds',video.h,acc_txt),video.x/2+ppd*1.2,video.y/2+prog_off+ppd*2.5); 
    Screen('TextSize',video.h,round(txtsiz));
    Screen('DrawText',video.h,acc_txt,acc_rec(1),acc_rec(2),0);
end

function draw_techn(itechn)
    % need to know which machine/bandit is in play to assign correct technician
        Screen('DrawTexture',video.h,techn_tex(itechn,1),[],techn_rec(1,:),[],[],[],0);
end

function draw_calib(icalib)
    if mod(icalib,2) == 0
        Screen('DrawTexture',video.h,calib_tex(1),[],calib_rec(1,:),[],[],[],0);
        Screen('DrawTexture',video.h,calib_tex(2),[],calib_rec(2,:),[],[],[],0);
    else
        Screen('DrawTexture',video.h,calib_tex(1),[],calib_rec(2,:),[],[],[],0);
        Screen('DrawTexture',video.h,calib_tex(2),[],calib_rec(1,:),[],[],[],0);
    end
end

function draw_stim
    ic = blck.color;
    % draw background rectangles
    Screen('FillRect', video.h, [1 0 0], bg_rect_frame_cntrd);
    Screen('FillRect', video.h, color_frame(ic,:), bg_rect_frame_cntrd);
    Screen('FillRect', video.h, color_rgb(ic,:), bg_rect_cntrd);
    % draw left stimulus
    is = blck.shape(il);
    Screen('DrawTexture',video.h,shape_tex(1,is),[],shape_rec(1,:),[],[],[],0);
    % draw right stimulus
    is = blck.shape(ir);
    Screen('DrawTexture',video.h,shape_tex(1,is),[],shape_rec(2,:),[],[],[],0);
end

function draw_levers(lever_color)  % 1-left,up; 2-left,down; 3-right,up; 4-right,down;
    if lever_up == true
        % draw both levers in up position
        Screen('DrawTexture',video.h,lever_tex(1),[],lever_rec(1,:),[],[],[],lever_color);
        Screen('DrawTexture',video.h,lever_tex(3),[],lever_rec(3,:),[],[],[],lever_color);
    else
        if key == 1 % left chosen
            % draw lever left in down
            Screen('DrawTexture',video.h,lever_tex(2),[],lever_rec(2,:),[],[],[],lever_color);
            % draw lever right in up
            Screen('DrawTexture',video.h,lever_tex(3),[],lever_rec(3,:),[],[],[],lever_color);
        else
            % draw lever left in up
            Screen('DrawTexture',video.h,lever_tex(1),[],lever_rec(1,:),[],[],[],lever_color);
            % draw lever right in down
            Screen('DrawTexture',video.h,lever_tex(4),[],lever_rec(4,:),[],[],[],lever_color);
        end
    end
end

function draw_feedback
    % draw the thicker bordered stimulus
    if key == 1 % left option
        %lr = -1;
        is = blck.shape(il);
        Screen('DrawTexture',video.h,shape_tex(2,is),[],shape_rec(1,:),[],[],[],0);
    else
        %lr = 1;
        is = blck.shape(ir);
        Screen('DrawTexture',video.h,shape_tex(2,is),[],shape_rec(2,:),[],[],[],0);
    end
    % text
    fb_txt = num2str(fb_val);
    Screen('TextSize',video.h,round(txtsiz*fbtxt_fac));
    %fb_rec = CenterRectOnPoint(Screen('TextBounds',video.h,fb_txt),video.x/2+(ppd*1.3*lr),video.y/2-shape_off/2); % old left-right feedback pos. 
    fb_rec = CenterRectOnPoint(Screen('TextBounds',video.h,fb_txt),video.x/2,video.y/2); 
    Screen('DrawText',video.h,fb_txt,fb_rec(1),fb_rec(2),.2);
end

function draw_feedback_trn(respsarr,pointsarr)
    Screen('TextSize',video.h,round(txtsiz*0.6));
    
    trn_bar = [0 0 ppd 1.5*ppd];    % vertex positions of bars (for sizing)
    trn_rec = zeros(ntrl,4);        % array for storing bar rectangle positions
    
    trn_points_txt = 'Les points que vous avez vu :';
    trn_choice_txt = 'Vos réponses :';
    
    % Create the bars and points and position them
    for iresp = 1:length(respsarr)
        if respsarr(iresp) == 1 % correct
            trn_bar_color = [1 1 1]; %white
            shape_chosen = 1;
            letter_color = 0;
        else
            trn_bar_color = [0 0 0]; %black
            shape_chosen = 2;
            letter_color = 1;
        end
        
        % Convert points values to text
        points_txt = num2str(pointsarr(iresp));
        
        % Position bars and points
        trn_rec(iresp,:) = CenterRectOnPoint(trn_bar,video.x/2+(iresp-(length(respsarr)/2+.5))*ppd*1.2,video.y/2-shape_off/2);
        points_rec       = CenterRectOnPoint(Screen('TextBounds',video.h,points_txt), ...
                           video.x/2+(iresp-(length(respsarr)/2+.5))*ppd*1.2,video.y/2-shape_off/2-1.5*ppd);
                     
        % Position descriptive text
        if iresp == 1
            points_txt_rec = [0 0 video.x/2+(iresp-(length(respsarr)/2)-1.5)*ppd*1.2 video.y];
            DrawFormattedText(video.h,trn_points_txt,'right',video.y/2-shape_off/2-1.5*ppd+4, 0,[],[],[],[],[],points_txt_rec);
            DrawFormattedText(video.h,trn_choice_txt,'right',video.y/2-shape_off/2+4, 0,[],[],[],[],[],points_txt_rec);
        end
        
        % Draw bars and points
        Screen('FillRect',video.h, trn_bar_color, trn_rec(iresp,:));
        Screen('DrawText',video.h,points_txt,points_rec(1),points_rec(2),0);
        
        % Draw the chosen letter superposed over the bars
        Screen('DrawTexture',video.h,trnfb_tex(1,blck.shape(shape_chosen)-6),[],trn_rec(iresp,:),[],[],[],letter_color);
    end
    
end

function load_instructions_pre1
    inst_txt{1} = 'Imaginez que vous êtes au casino.'; 
    inst_txt{2} = 'Vous allez jouer à 3 machines à sous :';
    inst_txt{3} = 'une orange, une bleue, et une grise.';
    inst_txt{4} = 'Chaque machine est associée à deux formes entre lesquelles';
    inst_txt{5} = 'vous devrez choisir en tirant sur le levier correspondant.';
    inst_txt{6} = 'Le jeu se divise en manches, c’est à dire';
    inst_txt{7} = 'un certain nombre de tirages d’une même machine.';
    inst_txt{8} = 'Vous changerez de machine à chaque nouvelle manche.';
    inst_txt{9} = 'Appuyez sur [espace] pour continuer.';
    
    itxt_end = 9;
    for itxt = 1:itxt_end
        if itxt<itxt_end
            txtppd = (itxt*ppd_mult-0.5)*ppd;
        else
            txtppd = (itxt*ppd_mult)*ppd;
        end
        
        inst_rec(itxt,:) = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt{itxt}),video.x/2,txtppd);
        Screen('DrawText',video.h,inst_txt{itxt},inst_rec(itxt,1),inst_rec(itxt,2),0);
    end
end

function load_instructions_pre2
    inst_txt{1}  = 'A chaque tirage, vous observerez le score associé';
    inst_txt{2}  = 'à la forme que vous avez choisie.';
    inst_txt{3}  = 'Ce score varie d’un tirage à l’autre,';
    inst_txt{4}  = 'mais l’une des deux formes est <i>statistiquement<i> meilleure que l’autre :'; % italics
    inst_txt{5}  = 'elle donne plus souvent des scores supérieurs à 50 points,';
    inst_txt{6}  = 'alors que l’autre forme donne plus souvent'; 
    inst_txt{7}  = 'des scores inférieurs à 50 points.';
    inst_txt{8}  = 'Le but du jeu est de choisir la meilleure forme,';
    inst_txt{9}  = 'même si elle donne parfois des scores inférieurs à 50 points.'; 
    inst_txt{10} = 'C’est le nombre de fois où vous choisirez la meilleure forme'; 
    inst_txt{11} = 'qui sera utilisé pour évaluer votre performance à l’issue'; 
    inst_txt{12} = 'de chaque manche, même si la forme en question a pu donner'; 
    inst_txt{13} = 'des scores inférieurs à 50 points à certains tirages.'; 
    inst_txt{14} = 'La meilleure forme ne change jamais au milieu d’une manche.'; 
    inst_txt{15} = 'Appuyez sur [espace] pour continuer.';
    
    itxt_end = 15;
    for itxt = 1:itxt_end
        if itxt<itxt_end
            txtppd = (itxt*ppd_mult-0.5)*ppd;
        else
            txtppd = (itxt*ppd_mult)*ppd;
        end
        
        inst_rec(itxt,:) = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt{itxt}),video.x/2,txtppd);
        if itxt == 4
            DrawFormattedText2(inst_txt{itxt}, 'win',video.h,'sx','center','sy','top','xalign','center','yalign','center',... 
                                               'transform',{'translate',[0 inst_rec(itxt,2)+ppd*0.5]});
        else
            Screen('DrawText',video.h,inst_txt{itxt},inst_rec(itxt,1),inst_rec(itxt,2),0);
        end
    end
end

function load_instructions_post1
    inst_txt{1}  = 'Comme vous avez pu le voir pendant l''entraînement,'; 
    inst_txt{2}  = 'la meilleure forme peut parfois donner un score inférieur à 50,';
    inst_txt{3}  = 'et la mauvaise forme peut parfois donner un score supérieur à 50.';
    inst_txt{4}  = 'Attention : ne vous laissez pas tromper par ces tirages !';
    inst_txt{5}  = '<b>La meilleure forme ne change jamais en cours de manche,<b>'; 
    inst_txt{6}  = '<b>et le but du jeu est de choisir la meilleure forme sélectionnée<b>';
    inst_txt{7}  = 'par le technicien pour la manche en cours -';
    inst_txt{8}  = 'pas celle qui a donné un score supérieur à 50 au tirage précédent.';
    inst_txt{9}  = '<b>C''est le nombre de fois où vous avez choisi la meilleure forme<b>'; 
    inst_txt{10} = '<b>qui sera utilisé pour évaluer vos performances,<b>';
    inst_txt{11} = 'pas la somme des scores affichés à l''écran.'; 
    inst_txt{12} = 'Appuyez sur [espace] pour continuer.';
    
    itxt_end = 12;
    for itxt = 1:itxt_end
        if itxt<itxt_end
            txtppd = (itxt*ppd_mult-0.5)*ppd;
        else
            txtppd = (itxt*ppd_mult)*ppd;
        end
        
        inst_rec(itxt,:) = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt{itxt}),video.x/2,txtppd);
        if ismember(itxt,[5 6 9 10])
            DrawFormattedText2(inst_txt{itxt}, 'win',video.h,'sx','center','sy','top','xalign','center','yalign','center',... 
                                               'transform',{'translate',[0 inst_rec(itxt,2)+ppd*0.5]});
        else
            Screen('DrawText',video.h,inst_txt{itxt},inst_rec(itxt,1),inst_rec(itxt,2),0);
        end
    end
end

function load_instructions_post2
    inst_txt{1}  = 'Les deux machines colorées sur lesquelles vous allez jouer'; 
    inst_txt{2}  = 'seront recalibrées par leur technicien avant chaque manche.';
    inst_txt{3}  = 'Chaque technicien est responsable de sa machine.';
    inst_txt{4}  = 'A chaque recalibration, le technicien va sélectionner laquelle';
    inst_txt{5}  = 'des deux formes sera la meilleure pour la manche qui va suivre.'; 
    inst_txt{6}  = 'Chaque technicien utilise sa propre stratégie pour recalibrer';
    inst_txt{7}  = 'sa machine avant chaque manche.';
    inst_txt{8}  = 'A vous de découvrir la stratégie utilisée par chaque technicien';
    inst_txt{9}  = 'pour essayer de deviner la forme qu’il a sélectionnée pour la manche'; 
    inst_txt{10} = 'qui va suivre. Vous pourrez ainsi choisir la meilleure forme';
    inst_txt{11} = 'plus souvent et augmenter votre performance.'; 
    inst_txt{12} = 'Appuyez sur [espace] pour continuer.';
    
    itxt_end = 12;
    for itxt = 1:itxt_end
        if itxt<itxt_end
            txtppd = (itxt*ppd_mult-0.5)*ppd;
        else
            txtppd = (itxt*ppd_mult)*ppd;
        end
        inst_rec(itxt,:) = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt{itxt}),video.x/2,txtppd);
        Screen('DrawText',video.h,inst_txt{itxt},inst_rec(itxt,1),inst_rec(itxt,2),0);
    end
end

function load_instructions_post3
    inst_txt{1}  = 'Un exemple de stratégie pourrait être de <b>toujours sélectionner<b>'; 
    inst_txt{2}  = '<b>la même forme comme la meilleure pour toutes les manches.<b>';
    inst_txt{3}  = 'Une autre stratégie possible pourrait être de <b>toujours changer<b>';
    inst_txt{4}  = '<b>de forme d''une manche à l''autre.<b>';
    inst_txt{5}  = 'A vous de découvrir la stratégie utilisée par chaque technicien'; 
    inst_txt{6}  = 'pour augmenter vos performances.';
    inst_txt{7}  = 'Attention : la machine grise change de formes à chaque';
    inst_txt{8}  = 'nouvelle manche, et aucun technicien ne la recalibre :';
    inst_txt{9}  = 'pour essayer de deviner la forme qu’il a sélectionnée pour la manche'; 
    inst_txt{10} = 'il n''y a donc aucune stratégie à découvrir pour cette machine !';
    inst_txt{11} = 'Appuyez sur [espace] pour continuer.';
    
    itxt_end = 11;
    for itxt = 1:itxt_end
        if itxt<itxt_end
            txtppd = (itxt*ppd_mult-0.5)*ppd;
        else
            txtppd = (itxt*ppd_mult)*ppd;
        end
        
        inst_rec(itxt,:) = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt{itxt}),video.x/2,txtppd);
        if ismember(itxt,1:4)
            DrawFormattedText2(inst_txt{itxt}, 'win',video.h,'sx','center','sy','top','xalign','center','yalign','center',... 
                                               'transform',{'translate',[0 inst_rec(itxt,2)+ppd*0.5]});
        else
            Screen('DrawText',video.h,inst_txt{itxt},inst_rec(itxt,1),inst_rec(itxt,2),0);
        end
    end
end

function key_training
    key_prac0l  = 'Touche';
    key_prac0r  = 'Touche';
    key_prac1   = 'E';
    key_prac2   = 'P';
    
    key_rec0l   = CenterRectOnPoint(Screen('TextBounds',video.h,key_prac0l),video.x/2-shape_off*2.2,video.y/2);
    key_rec0r   = CenterRectOnPoint(Screen('TextBounds',video.h,key_prac0r),video.x/2+shape_off*2.3,video.y/2);
    key_rec1    = CenterRectOnPoint(Screen('TextBounds',video.h,key_prac1),video.x/2-shape_off*2.2,video.y/2+ppd*2.0);
    key_rec2    = CenterRectOnPoint(Screen('TextBounds',video.h,key_prac2),video.x/2+shape_off*2.2,video.y/2+ppd*2.0);
    
    Screen('TextSize',video.h,round(txtsiz)*1.3);
    Screen('DrawText',video.h,key_prac0l,key_rec0l(1),key_rec0l(2),0);
    Screen('DrawText',video.h,key_prac0r,key_rec0r(1),key_rec0r(2),0);
    Screen('DrawText',video.h,key_prac1,key_rec1(1),key_rec1(2),0);
    Screen('DrawText',video.h,key_prac2,key_rec2(1),key_rec2(2),0);
end

function load_pre_prac   
    prac_txt{1}  = 'Vous allez commencer par jouer à quelques manches pour';
    prac_txt{2}  = 'vous entraîner, sur une autre machine où les formes';
    prac_txt{3}  = 'sont remplacées par des lettres qui changeront à chaque manche.';
    prac_txt{4}  = 'Le but de cet entraînement est de vous familiariser'; 
    prac_txt{5}  = 'avec le fonctionnement du jeu.';
    prac_txt{6}  = 'Par exemple, comme vous pourrez le voir, la position des lettres';
    prac_txt{7}  = '(et donc des formes dans les manches qui vont suivre)';
    prac_txt{8}  = 'peut changer d’un tirage à l’autre.';
    prac_txt{9}  = 'Faites donc attention à tirer le levier correspondant'; 
    prac_txt{10} = 'à la lettre (ou la forme) que vous souhaitez choisir.'; 
    prac_txt{11} = 'C’est parti !';
    prac_txt{12} = 'Appuyez sur [espace] pour continuer.';

    itxt_end = 12;
    for itxt = 1:itxt_end
        if itxt<itxt_end-1
            txtppd = (itxt*ppd_mult-0.5)*ppd;
        elseif itxt == itxt_end-1
            txtppd = (itxt*ppd_mult+1.0)*ppd;
        else
            txtppd = (itxt*ppd_mult+1.0)*ppd;
        end
        
        prac_rec(itxt,:) = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt{itxt}),video.x/2,txtppd);
        Screen('DrawText',video.h,prac_txt{itxt},prac_rec(itxt,1),prac_rec(itxt,2),0);
    end
end

function missedpupil = check_pupil(trackeye,eye_used)
% check for good positioning of pupil
    if trackeye
        pupilthere = false;
        while ~pupilthere
            % checks if new (float) sample is available
            % returns -1 if none or error, 0 if old, 1 if new
            if Eyelink('NewFloatSampleAvailable') > 0
                evt = Eyelink( 'NewestFloatSample');
                eyeposx = evt.gx(eye_used+1);
                eyeposy = evt.gy(eye_used+1);
                if eyeposx~=el.MISSING_DATA && eyeposy~=el.MISSING_DATA % if valid
                    pupilthere = 1;
                    missedpupil = 0;
                else % if invalid
                    missedpupil = 1;
                    pupiltxt{1} = '-Erreur d''enregistrement de pupille-';
                    pupiltxt{2} = 'Repositionnez votre tête dans la bonne position /';
                    pupiltxt{3} = 'Gardez les yeux bien ouverts.';
                    pupiltxt{4} = 'Appuyez sur [espace] pour continuer.';
                    pupiltxt{5} = 'Si le jeu ne continue pas après avoir appuyé [espace],';
                    pupiltxt{6} = 'votre tête n''est pas dans la bonne position.';
                    pupiltxt{7} = 'Dans le cas où le jeu ne commence pas,';
                    pupiltxt{8} = 'veuillez appelez l''expérimentateur.';
                    
                    pupilrec = zeros(6,4);
                    for itxt = 1:8
                        if itxt < 4
                            txtppd = ((itxt-4)*ppd_mult)*ppd;
                        else
                            txtppd = ((itxt-3)*ppd_mult)*ppd;
                        end
                        pupilrec(itxt,:) = CenterRectOnPoint(Screen('TextBounds',video.h,pupiltxt{itxt}),video.x/2,video.y/2+txtppd);
                        Screen('DrawText',video.h,pupiltxt{itxt},pupilrec(itxt,1),pupilrec(itxt,2),0);
                    end
                    Screen('DrawingFinished',video.h);
                    Screen('Flip',video.h);
                    WaitKeyPress(keywait,[],false); % press space to move on
                    pause(.2);
                    
                    % check if pupil is in fact in the right position
                    if Eyelink('NewFloatSampleAvailable')>0
                        evt = Eyelink( 'NewestFloatSample');
                        eyeposx = evt.gx(eye_used+1);
                        eyeposy = evt.gy(eye_used+1);
                        if eyeposx~=el.MISSING_DATA && eyeposy~=el.MISSING_DATA
                            pupilthere = 1;
                        else
                            continue;
                        end
                    end
                end
            end
        end
    end
end

end % main function
