function [expe,aborted,errmsg] = run_expe_rlvsl(subj,syncflip,trackeye)

% Advisory: Before running the experiment on actual participants, be sure to go
%           through the code searching for 'debug' in any commented sections. These must either be
%           re-enabled or disabled for proper functionality and data collection - J Lee.

% toggle fastmode       to go through the experiment faster but still at a reasonable pace
% toogle superfastmode  to go through the experiment at lightning speed
fastmode        = true; % debug
superfastmode   = true; % debug

if fastmode | superfastmode
    beep on;
    for i = 1:10
        disp('Fast mode or Superfast mode is turned on! Abandon experiment if testing real participants!!');
        beep;
        pause(.1);
    end
end

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

% create header
hdr = [];
hdr.subj = subj;
hdr.date = datestr(now,'yyyymmdd-HHMM');

% load experiment structure
filename = sprintf('./Data/S%02d/RLVSL_S%02d_expe.mat',subj,subj)
% filename = strcat(pwd,'\Data\RLVAR_S0',num2str(subj),'_expe.mat'); 
if ~exist(filename,'file')
    error('Missing experiment file!');
end
load(filename,'expe');
expe(1).hdr = hdr;

% set color-technician pairings
techn_order = set_techn(mod(subj,6));
expe(1).techn_order = techn_order;

% define output arguments
aborted = false; % aborted prematurely?
errmsg = []; % error message

% get and set screen parameters
argiscr = inputdlg({'Select screen (0:main, 1:extended)'},'RLVSL',1,{'','','','',''});
if ~ismember(argiscr,{'0','1'})
    error('Unexpected input. Enter either 0 or 1.');
end
iscr = str2num(argiscr{1}); % screen index (change to 0 if no extended screens)
res  = []; % screen resolution
fps  = []; % screen refresh rate
ppd  = 40; % number of screen pixels per degree of visual angle

% set list of color-wise R/G/B values
color_rgb   = [ ...  % inner rectangle
    251,229,214; ... % red
    226,240,217; ... % green
    222,235,247; ... % blue
    192,192,192; ... % grey
    ]/255;
color_frame = [ ...  % darker outside border rectangle
    244,177,131; ...
    169,209,142; ...
    157,195,230; ...
    100,100,100; ...
    ]/255;

% set stimulation parameters
lumibg    = 128/255; % background luminance
fixtn_siz = 0.3*ppd; % fixation point size
shape_siz = 4.0*ppd; % shape size
tools_siz = 2.0*ppd; % tools and gears size
shape_off = 4.0*ppd; % shape offset
fbtxt_fac = 1.5;     % feedback text magnification factor
intxt_fac = 1.5;     % instruction text magnification factor
prog_off    = ppd*8.0; % bonus texture offset
prog1_xoff	= ppd*3.8;
prog2_xoff  = ppd*7.7;

% create video structure
video = [];

try
    % hide cursor and prevent spilling key presses into MATLAB windows
    %HideCursor;  % debug (enable when finished debugging)
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
    keyresp = KbName({'Q','P'});
    
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
        txtsiz = round(1.0*ppd);
    elseif ispc
        txtfnt = 'Arial'; % closest to Helvetica
        txtsiz = round(2/3*ppd); % text size is ~2/3 smaller in Windows than MacOSX
    end
    
    % set font properties
    if ismac
        txtfnt = 'Helvetica';
        txtsiz = round(1.0*ppd);
    elseif ispc
        txtfnt = 'Arial'; % closest to Helvetica
        txtsiz = round(2/3*ppd); % text size is ~2/3 smaller in Windows than MacOSX
    end
    
    % PsychToolbox properties
    Screen('Preference','TextAlphaBlending',1);
    Screen('Preference','DefaultFontName',txtfnt);
    Screen('Preference','DefaultFontSize',txtsiz);
    Screen('Preference','DefaultFontStyle',0);
    %Screen('Preference', 'SkipSyncTests', 1);  %%% disabled for testing - JL debug
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
    
    % ******** initialize eye-tracker connection ************** not included yet
    
    if true % shape loading code (coded for reducing clutter)   
    % Note: The PNG files must be in GREYSCALE format
    
    % Load technician shape textures
    techn_tex = zeros(3,2); % techn_tex(1,:)=Alice, techn_tex(2,:)=Bob, techn_tex(3,:)=Charlie
    for it = 1:3
        img     = double(imread(sprintf('./img/techn%d.png',it)))/255;  % standard technician avatar
        img     = imresize(img,shape_siz/size(img,1));
        techn_tex(it,1) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
        img     = double(imread(sprintf('./img/techn%dw.png',it)))/255; % winking technician avatar
        img     = imresize(img,shape_siz/size(img,1));
        techn_tex(it,2) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    % Position technician shape textures
    techn_rec = zeros(1,4);
    techn_rec = CenterRectOnPoint(Screen('Rect',techn_tex(1,1)),video.x/2,video.y/2-shape_off*1.55);
    
    % Load tools/gears shape textures
    calib_tex = zeros(2,1); % calib_tex(1)=tools, calib_tex(2)=gears
    for it = 1:2
        img     = double(imread(sprintf('./img/calib%d.png',it)))/255;
        img     = imresize(img,tools_siz/size(img,1));
        calib_tex(it,1) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    % Position tools/gears shape textures
    calib_rec       = zeros(2,4); % calib_rec(1,:)=left, calib_rec(2,:)=right
    calib_rec(1,:)  = CenterRectOnPoint(Screen('Rect',calib_tex(1,1)),video.x/2-shape_off/1.3,video.y/2-shape_off*2); % left
    calib_rec(2,:)  = CenterRectOnPoint(Screen('Rect',calib_tex(1,1)),video.x/2+shape_off/1.3,video.y/2-shape_off*2); % right
    
    % Load bandit shape textures
    shape_tex = zeros(2,8); 
    for is = 1:8
        img     = double(imread(sprintf('./img/shape%d.png',is)))/255;  % unchosen stimuli
        img     = imresize(img,shape_siz/size(img,1));
        shape_tex(1,is) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
        img     = double(imread(sprintf('./img/shape%dc.png',is)))/255; % chosen stimuli
        img     = imresize(img,shape_siz/size(img,1));
        shape_tex(2,is) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    % Position bandit shape textures
    shape_rec       = zeros(2,4); % shape_rec(1,:)=left shape_rec(2,:)=right
    shape_rec(1,:)  = CenterRectOnPoint(Screen('Rect',shape_tex(1,1)),video.x/2-shape_off,video.y/2-shape_off/2); % left
    shape_rec(2,:)  = CenterRectOnPoint(Screen('Rect',shape_tex(1,1)),video.x/2+shape_off,video.y/2-shape_off/2); % right
    
    % Create background rectangles and outer frames
    bg_rect             = [video.x/2-(1.5*shape_off) video.y/2-shape_off/2 video.x/2+(1.5*shape_off) video.y/2+shape_off/2];
    bg_rect_frame       = [video.x/2-(1.55*shape_off) video.y/2-shape_off/1.8 video.x/2+(1.55*shape_off) video.y/2+shape_off/1.8];
    bg_rect_cntrd       = CenterRectOnPoint(bg_rect,video.x/2,video.y/2-shape_off/2);
    bg_rect_frame_cntrd = CenterRectOnPoint(bg_rect_frame,video.x/2,video.y/2-shape_off/2);
    
    % Load lever textures
    lever_tex = zeros(4,1); % 1-left,up; 2-left,down; 3-right,up; 4-right,down;
    for is = 1:4
        img = double(imread(sprintf('./img/lever%d.png',is)))/255; 
        img = imresize(img,shape_siz/size(img,1));
        lever_tex(is,1) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    % Position lever textures
    lever_rec = zeros(2,4);
    lever_rec(1,:) = CenterRectOnPoint(Screen('Rect',lever_tex(1,1)),video.x/2-(1.55*shape_off)-87,video.y/2-shape_off/2);
    lever_rec(2,:) = CenterRectOnPoint(Screen('Rect',lever_tex(2,1)),video.x/2+(1.55*shape_off)+86,video.y/2-shape_off/2); 
    
    % Load progress bar textures
    prog_tex = zeros(1,9);
    for ib = 1:9
        img = double(imread(sprintf('./img/level%d.png',ib)))/255;
        img = imresize(img,shape_siz/size(img,1));
        prog_tex(1,ib) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    % Position progress bar textures
    prog_rec      = zeros(9,4);
    prog_tex_left = zeros(1,2);
    for ib = 1:2
        img = double(imread(sprintf('./img/bonus%d.png',ib)))/255;
        img = imresize(img,shape_siz/size(img,1));
        prog_tex_left(1,ib) = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    end
    prog_rec_left  = zeros(2,4);
    
    end % end of texture loading and positioning code
    
    % create fixation point
    img         = CreateCircularAperture(fixtn_siz);
    fixtn_tex   = Screen('MakeTexture',video.h,cat(3,ones(size(img)),img),[],[],2);
    fixtn_rec   = CenterRectOnPoint(Screen('Rect',fixtn_tex(1)),video.x/2,video.y/2-shape_off/2);
    
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
        
        ntrl = length(expe(iblk).blck);
        
        % initialize response structure
        resp    = zeros(1,ntrl);    % response (1:correct, 2:incorrect)
        rt      = zeros(1,ntrl);    % response time (seconds)
        
        % draw start screen
        Screen('TextStyle',video.h,0);
        Screen('TextSize',video.h,txtsiz);
        
        if isprac(iblk)
            if iblk == 1
                % load instructions screen
                load_instructions;
                labeltxt = 'Press [space] to continue';
                labelrec = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt),video.x/2,18.0*ppd);
                Screen('DrawText',video.h,labeltxt,labelrec(1),labelrec(2),0);
                Screen('Flip',video.h);
                WaitKeyPress(keywait,[],false); % press space to move on
                
                % load pre-practice screen
                load_pre_prac;
                labeltxt = 'Press [space] to begin your training';
                labelrec = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt),video.x/2,18.0*ppd);
                Screen('DrawText',video.h,labeltxt,labelrec(1),labelrec(2),0);
                Screen('Flip',video.h);
                pause(2);
                WaitKeyPress(keywait,[],false); % press space to move on to next block
            end
            disp(['For proctor purposes: ' num2str(blck.shape)]);
        else % real blocks
            if iblk == nblk_prac + 1
                labeltxt = 'Press [space] to begin the real sessions';
                labelrec = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt),video.x/2,round(1.2*ppd));
                Screen('DrawText',video.h,labeltxt,labelrec(1),labelrec(2),0);
                Screen('TextSize',video.h,round(txtsiz*intxt_fac));
                Screen('Flip',video.h);
                pause(2);
                WaitKeyPress(keywait,[],false); % press space to move on to next block
            end
            
            % display technician calibrating machine
            if iblk < nblk
                ic_calib = expe(iblk).color;
                for icalib = 1:4
                    % block appropriate technician avatar
                    itechn = techn_order(ic_calib);
                    draw_techn(itechn,false); % normal technician avatar
                    Screen('FillRect', video.h, color_frame(ic_calib,:), bg_rect_frame_cntrd);
                    Screen('FillRect', video.h, color_rgb(ic_calib,:), bg_rect_cntrd);
                    draw_calib(icalib);
                    Screen('DrawingFinished',video.h);
                    Screen('Flip',video.h);
                    pause(.3);
                end
                pause(.2);
                draw_techn(itechn,true); % winking technician avatar
                Screen('FillRect', video.h, color_frame(ic_calib,:), bg_rect_frame_cntrd);
                Screen('FillRect', video.h, color_rgb(ic_calib,:), bg_rect_cntrd);
                draw_calib(icalib);
                labeltxt = 'Press [space] to play.';
                labelrec = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt),video.x/2,20.0*ppd);
                Screen('DrawText',video.h,labeltxt,labelrec(1),labelrec(2),0);
                Screen('DrawingFinished',video.h);
                Screen('Flip',video.h);
                pause(2);
                WaitKeyPress(keywait,[],false); % press space to move on to next block
            end
        end
        
        % draw fixation point
        Screen('DrawTexture',video.h,fixtn_tex,[],fixtn_rec,[],[],[],0);
        Screen('DrawingFinished',video.h);
        t = Screen('Flip',video.h);
        
        for itrl = 1:ntrl
            
            if CheckKeyPress(keyquit) % check for abort
                aborted = true;
                break;
            end
            
            lever_up = true; % status of lever positions (up)
            % place correct shape in its randomly chosen position
            if blck.locat(itrl) == 1 % if correct place is on the left
                il = 1; % left position gets correct shape
                ir = 2;
            else                     % if correct place is on the right
                il = 2; % right position gets correct shape
                ir = 1;
            end
            draw_stim;
            Screen('DrawTexture',video.h,fixtn_tex,[],fixtn_rec,[],[],[],0);
            draw_levers;
            
            if fastmode || superfastmode
                t = Screen('Flip',video.h,t+roundfp(0.000,0.000));  % trigger only when testing
            else
                t = Screen('Flip',video.h,t+roundfp(2.000,0.500));  % for the real experimental situation
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
            
            % calculate feedback (value)
            ease_mult = 1.5;    % how much easier the training blocks should be in terms of mean difference multiple
            mu_new  = 60;       % desired mean of distribution
            if isprac   % remapping of value difference for practice blocks
                mu_new  = 50+(10*ease_mult);   % desired mean of distribution
            end
            sig_new = 15;   % desired std of distribution
            a = sig_new/expe(1).cfg.sgen;       % slope of linear transformation aX+b
            b = mu_new - a*expe(1).cfg.mgen;    % intercept of linear transf. aX+b
            
            fb_val = blck.blck(itrl);
            blck.blck(itrl) = round(fb_val*a+b); % unsigned transformed value to store in expe structure
            if resp(itrl) ~= 1
                fb_val = fb_val*-1; % sign the value based on choice
            end
            % rescale random variable distr. from N(.1,.2^2) to N(60,15^2)
            fb_val = round(fb_val*a+b);
            % clip extreme values
            if fb_val > 99
                fb_val = 99;
            elseif fb_val < 1
                fb_val = 1;
            end
            
            % show feedback
            draw_stim;
            draw_levers;
            draw_feedback;
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h);
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
            Screen('Flip',video.h);
            
            
        end % end of trial loop
        
        acc_score = [acc_score; resp]; % store accuracy scores for each block in array
            
        % update expe struct.
        expe(iblk).resp     = resp;
        expe(iblk).rt       = rt;
        expe(iblk).blck_trn = blck.blck;
        
        if aborted
            break;
        end
        
        % end of training blocks
        if iblk < nblk_prac+1
            draw_feedback_trn(resp);
            labeltxt1 = 'The good letter was:';
            trn_shape_rec = CenterRectOnPoint(Screen('Rect',shape_tex(1,1)),video.x/2,6.0*ppd); % center good letter
            labeltxt2 = sprintf('Your accuracy this round was: %.1f%%', numel(resp(resp==1))/ntrl*100); % accuracy for that block
            labeltxt3 = '(Black bars represent when you made a correct choice)';
            labeltxt4 = 'Press [space] to continue.';
            
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
            Screen('DrawTexture',video.h,shape_tex(1,blck.shape(1)),[],trn_shape_rec,[],[],[],0); % draw good letter
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),0);
            Screen('DrawText',video.h,labeltxt4,labelrec4(1),labelrec4(2),0);
            
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h);
            pause(1);
            WaitKeyPress(keywait,[],false);
        end
        
        % end of block screen
        if iblk == nblk_prac % end of training session screen
            labeltxt1 = 'End of TRAINING sessions.';
            labeltxt2 = 'Press [space] to continue to the real sessions.';
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,video.y/2-3.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2+3.0*ppd);
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),0);
            %draw_progress(expe(iblk).sesh);
            acc = calc_acc;
            draw_acc;
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h);
            pause(1);
            WaitKeyPress(keywait,[],false);
            
            load_instructions2;
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h);
            pause(1);
            WaitKeyPress(keywait,[],false);
            
            load_instructions3;
            
        elseif iblk == length(expe) % end of last session
            labeltxt1 = 'End of experiment.';
            labeltxt2 = 'Thanks for playing.';
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,video.y/2-3.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2+3.0*ppd);
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),0);
%            draw_progress(expe(iblk).sesh);
            acc = calc_acc;
            draw_acc;
        elseif mod(expe(iblk).sesh,2) == 1 && mod(iblk-nblk_prac,(length(expe)-nblk_prac)/expe(length(expe)).sesh)==0 % end of a session but not break time (odd)
            labeltxt1 = sprintf('End of session %d.',expe(iblk).sesh);
            labeltxt2 = 'Winning points is hard work!';
            labeltxt3 = 'Take a short break to rest your eyes and neck.';
            labeltxt4 = 'Press [space] to continue to next session.';
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,video.y/2-3.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2-1.0*ppd);
            labelrec3 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt3),video.x/2,video.y/2+3.0*ppd);
            labelrec4 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt4),video.x/2,video.y/2+5.0*ppd);
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),0);
            Screen('DrawText',video.h,labeltxt3,labelrec3(1),labelrec3(2),0);
            Screen('DrawText',video.h,labeltxt4,labelrec4(1),labelrec4(2),0);
%            draw_progress(expe(iblk).sesh);
            acc = calc_acc;
            draw_acc;
        elseif mod(expe(iblk).sesh,2) == 0 && mod(iblk-nblk_prac,(length(expe)-nblk_prac)/expe(length(expe)).sesh)==0 % break time screen (even)
            labeltxt1 = sprintf('End of session %d.',expe(iblk).sesh);
            labeltxt2 = 'Do not continue.';
            labeltxt3 = 'Notify the proctor.';
            labelrec1 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt1),video.x/2,video.y/2-4.0*ppd);
            labelrec2 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt2),video.x/2,video.y/2);
            labelrec3 = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt3),video.x/2,video.y/2+4.0*ppd);
            Screen('DrawText',video.h,labeltxt1,labelrec1(1),labelrec1(2),0);
            Screen('DrawText',video.h,labeltxt2,labelrec2(1),labelrec2(2),1);
            Screen('DrawText',video.h,labeltxt3,labelrec3(1),labelrec3(2),0);
%            draw_progress(expe(iblk).sesh);
            acc = calc_acc;
            draw_acc;
        else % not end of session
            labeltxt = 'Press [space] to continue.';
            labelrec = CenterRectOnPoint(Screen('TextBounds',video.h,labeltxt),video.x/2,round(1.2*ppd));
            Screen('DrawText',video.h,labeltxt,labelrec(1),labelrec(2),0);
        end
        Screen('TextSize',video.h,round(txtsiz*intxt_fac));
        Screen('DrawingFinished',video.h);
        Screen('Flip',video.h);
        pause(.5);
        WaitKeyPress(keywait,[],false); % press space to move on to next block
        
    end % end of block loop
    
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

function draw_progress(isesh) % disabled in main loop - JL
    ilevel = isesh + 1;
    % levels 1-3 are centered
    % levels 4-6 have bonus 1 to the left
    % levels 7-10 have bonus 1 and 2 to the left
    if ilevel < 4 
        % progress textures positioning
        prog_rec(ilevel,:) = CenterRectOnPoint(Screen('Rect',prog_tex(1,ilevel)),video.x/2,video.y/2+prog_off);
        Screen('DrawTexture',video.h,prog_tex(1,ilevel),[],prog_rec(ilevel,:),[],[],[],1);
    elseif ilevel < 7
        % left progress bar
        prog_rec_left(1,:)  = CenterRectOnPoint(Screen('Rect',prog_tex_left(1,1)),video.x/2-prog1_xoff,video.y/2+prog_off);
        Screen('DrawTexture',video.h,prog_tex_left(1,1),[],prog_rec_left(1,:),[],[],[],1);
        % right progress bar
        prog_rec(ilevel,:) = CenterRectOnPoint(Screen('Rect',prog_tex(1,ilevel)),video.x/2+prog1_xoff,video.y/2+prog_off);
        Screen('DrawTexture',video.h,prog_tex(1,ilevel),[],prog_rec(ilevel,:),[],[],[],.75);
    else
        % left progress bar
        prog_rec_left(1,:)  = CenterRectOnPoint(Screen('Rect',prog_tex_left(1,1)),video.x/2-prog2_xoff,video.y/2+prog_off);
        Screen('DrawTexture',video.h,prog_tex_left(1,1),[],prog_rec_left(1,:),[],[],[],1);
        % center progress bar
        prog_rec_left(2,:)  = CenterRectOnPoint(Screen('Rect',prog_tex_left(1,2)),video.x/2,video.y/2+prog_off);
        Screen('DrawTexture',video.h,prog_tex_left(1,2),[],prog_rec_left(2,:),[],[],[],.75); 
        % right progress bar
        prog_rec(ilevel,:) = CenterRectOnPoint(Screen('Rect',prog_tex(1,ilevel)),video.x/2+prog2_xoff+ppd*0.7,video.y/2+prog_off);
        Screen('DrawTexture',video.h,prog_tex(1,ilevel),[],prog_rec(ilevel,:),[],[],[],.25);
    end
end

function draw_techn(itechn,iswink)
    % need to know which machine/bandit is in play to assign correct technician
    if iswink
        Screen('DrawTexture',video.h,techn_tex(itechn,2),[],techn_rec(1,:),[],[],[],0); % change second index of techn_tex later
    else
        Screen('DrawTexture',video.h,techn_tex(itechn,1),[],techn_rec(1,:),[],[],[],0);
    end
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

function draw_levers  % 1-left,up; 2-left,down; 3-right,up; 4-right,down;
    if lever_up == true
        % draw both levers in up position
        Screen('DrawTexture',video.h,lever_tex(1),[],lever_rec(1,:),[],[],[],0);
        Screen('DrawTexture',video.h,lever_tex(3),[],lever_rec(2,:),[],[],[],0);
    else
        if key == 1 % left chosen
            % draw lever left in down
            Screen('DrawTexture',video.h,lever_tex(2),[],lever_rec(1,:),[],[],[],0);
            % draw lever right in up
            Screen('DrawTexture',video.h,lever_tex(3),[],lever_rec(2,:),[],[],[],0);
        else
            % draw lever left in up
            Screen('DrawTexture',video.h,lever_tex(1),[],lever_rec(1,:),[],[],[],0);
            % draw lever right in down
            Screen('DrawTexture',video.h,lever_tex(4),[],lever_rec(2,:),[],[],[],0);
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
    fb_rec = CenterRectOnPoint(Screen('TextBounds',video.h,fb_txt),video.x/2,video.y/2-shape_off/2); 
    Screen('DrawText',video.h,fb_txt,fb_rec(1),fb_rec(2),.2);
end

function draw_feedback_trn(respsarr)
    trn_bar = [0 0 ppd 1.5*ppd];
    trn_rec = zeros(ntrl,4); % array for storing bar rectangle positions
    
    for iresp = 1:length(respsarr)
        if respsarr(iresp) == 1 % correct
            trn_bar_color = [0 0 0]; %black
        else
            trn_bar_color = [1 1 1]; %white
        end
        trn_rec(iresp,:) = CenterRectOnPoint(trn_bar,video.x/2+(iresp-(length(respsarr)/2+.5))*ppd*1.2,video.y/2-shape_off/2);
        Screen('FillRect', video.h, trn_bar_color, trn_rec(iresp,:));
    end
    
end

function load_instructions
    inst_txt1 = 'Thank you for participating in our experiment.';
    inst_txt2 = 'In this experiment, you will play a game where the goal is';
    inst_txt3 = 'to guess which one of two presented shapes or letters';
    inst_txt4 = 'provides more points on average (more meaning greater than 50).';
    inst_txt5 = 'Imagine you have a 2-choice slot machine in front of you,';
    inst_txt6 = 'and there is one that is objectively better than the other';
    inst_txt7 = 'Once you know that one of the shapes is the best,';
    inst_txt8 = 'you wouldn''t care to try the other since it might''ve been an';
    inst_txt9 = 'unlucky pull.';
    inst_txt10 = 'It is the same with our game.';
    
    inst_rec1 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt1),video.x/2,1.0*ppd);
    inst_rec2 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt2),video.x/2,3.0*ppd);
    inst_rec3 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt3),video.x/2,4.5*ppd);
    inst_rec4 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt4),video.x/2,6.0*ppd);
    inst_rec5 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt5),video.x/2,8.0*ppd);
    inst_rec6 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt6),video.x/2,9.5*ppd);
    inst_rec7 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt7),video.x/2,11.0*ppd);
    inst_rec8 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt8),video.x/2,12.5*ppd);
    inst_rec9 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt9),video.x/2,14.0*ppd);
    inst_rec10 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt10),video.x/2,16.0*ppd);
    
    Screen('DrawText',video.h,inst_txt1,inst_rec1(1),inst_rec1(2),0);
    Screen('DrawText',video.h,inst_txt2,inst_rec2(1),inst_rec2(2),0);
    Screen('DrawText',video.h,inst_txt3,inst_rec3(1),inst_rec3(2),0);
    Screen('DrawText',video.h,inst_txt4,inst_rec4(1),inst_rec4(2),0);
    Screen('DrawText',video.h,inst_txt5,inst_rec5(1),inst_rec5(2),0);
    Screen('DrawText',video.h,inst_txt6,inst_rec6(1),inst_rec6(2),0);
    Screen('DrawText',video.h,inst_txt7,inst_rec7(1),inst_rec7(2),0);
    Screen('DrawText',video.h,inst_txt8,inst_rec8(1),inst_rec8(2),0);
    Screen('DrawText',video.h,inst_txt9,inst_rec9(1),inst_rec9(2),0);
    Screen('DrawText',video.h,inst_txt10,inst_rec10(1),inst_rec10(2),0);
end
function load_instructions2
    inst_txt1 = 'Imagine now that you are at the casino.';
    inst_txt2 = 'In front of you are 3 slot machines, each marked by a color';
    inst_txt3 = 'and a unique set of shapes for each machine.';
    inst_txt4 = 'You will pull on one machine for 16 pulls, and then have to move on';
    inst_txt5 = 'to another.';
    inst_txt6 = 'Each machine is calibrated by 1 of 3 technicians:'; 
    inst_txt7 = 'Alice, Bob, and Charlie.';
    % insert picture of their faces here in main loop
    inst_txt8 = 'During the task you''ll see which technician calibrates which machine.';
    inst_txt9 = 'Press [space] to continue.';
    
    inst_rec1 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt1),video.x/2,1.0*ppd);
    inst_rec2 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt2),video.x/2,3.0*ppd);
    inst_rec3 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt3),video.x/2,4.5*ppd);
    inst_rec4 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt4),video.x/2,6.0*ppd);
    inst_rec5 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt5),video.x/2,7.5*ppd);
    inst_rec6 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt6),video.x/2,9.0*ppd);
    inst_rec7 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt7),video.x/2,10.5*ppd);
    inst_rec8 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt8),video.x/2,20.0*ppd);
    inst_rec9 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt9),video.x/2,22.0*ppd);
    
    Screen('DrawText',video.h,inst_txt1,inst_rec1(1),inst_rec1(2),0);
    Screen('DrawText',video.h,inst_txt2,inst_rec2(1),inst_rec2(2),0);
    Screen('DrawText',video.h,inst_txt3,inst_rec3(1),inst_rec3(2),0);
    Screen('DrawText',video.h,inst_txt4,inst_rec4(1),inst_rec4(2),0);
    Screen('DrawText',video.h,inst_txt5,inst_rec5(1),inst_rec5(2),0);
    Screen('DrawText',video.h,inst_txt6,inst_rec6(1),inst_rec6(2),0);
    techn_instr_rec = CenterRectOnPoint(Screen('Rect',techn_tex(1,1)),video.x/2-4.0*ppd,14.5*ppd);
    Screen('DrawTexture',video.h,techn_tex(1,1),[],techn_instr_rec,[],[],[],0); % Alice
    techn_instr_rec = CenterRectOnPoint(Screen('Rect',techn_tex(2,1)),video.x/2,14.5*ppd);
    Screen('DrawTexture',video.h,techn_tex(2,1),[],techn_instr_rec,[],[],[],0); % Bob
    techn_instr_rec = CenterRectOnPoint(Screen('Rect',techn_tex(3,1)),video.x/2+4.0*ppd,14.5*ppd);    
    Screen('DrawTexture',video.h,techn_tex(3,1),[],techn_instr_rec,[],[],[],0); % Charlie
    Screen('DrawText',video.h,inst_txt7,inst_rec7(1),inst_rec7(2),0);
    Screen('DrawText',video.h,inst_txt8,inst_rec8(1),inst_rec8(2),0);
    Screen('DrawText',video.h,inst_txt9,inst_rec9(1),inst_rec9(2),0);
end

function load_instructions3
    inst_txt8 = 'At each calibration, the technician will set'; 
    inst_txt9 = 'the good shape for that round.';
    inst_txt10 = 'Each technician employs a different strategy';
    inst_txt11 = 'in setting the good shape.';
    inst_txt12 = 'Your goal is to choose the good shape set by the technician,'; 
    inst_txt13 = 'since that is how you will earn more points.';
    inst_txt14 = 'Remember: The outcome shown is there to help, but sometimes,';
    inst_txt15 = 'confuse you in determining the good shape.';
    inst_txt16 = '(i.e. the good shape can have outcomes less than 50,'; 
    inst_txt17 = 'and the bad shape can have outcomes more than 50';
    inst_txt18 = 'Press [space] to continue.';
    
    inst_rec8 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt8),video.x/2,1.0*ppd);
    inst_rec9 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt9),video.x/2,2.5*ppd);
    inst_rec10 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt10),video.x/2,4.5*ppd);
    inst_rec11 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt11),video.x/2,6.0*ppd);
    inst_rec12 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt12),video.x/2,8.0*ppd);
    inst_rec13 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt13),video.x/2,9.5*ppd);
    inst_rec14 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt14),video.x/2,11.5*ppd);
    inst_rec15 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt15),video.x/2,13.0*ppd);
    inst_rec16 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt16),video.x/2,14.5*ppd);
    inst_rec17 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt17),video.x/2,16.0*ppd);
    inst_rec18 = CenterRectOnPoint(Screen('TextBounds',video.h,inst_txt18),video.x/2,18.0*ppd);
    
    Screen('DrawText',video.h,inst_txt8,inst_rec8(1),inst_rec8(2),0);
    Screen('DrawText',video.h,inst_txt9,inst_rec9(1),inst_rec9(2),0);
    Screen('DrawText',video.h,inst_txt10,inst_rec10(1),inst_rec10(2),0);
    Screen('DrawText',video.h,inst_txt11,inst_rec11(1),inst_rec11(2),0);
    Screen('DrawText',video.h,inst_txt12,inst_rec12(1),inst_rec12(2),0);
    Screen('DrawText',video.h,inst_txt13,inst_rec13(1),inst_rec13(2),0);
    Screen('DrawText',video.h,inst_txt14,inst_rec14(1),inst_rec14(2),0);
    Screen('DrawText',video.h,inst_txt15,inst_rec15(1),inst_rec15(2),0);
    Screen('DrawText',video.h,inst_txt16,inst_rec16(1),inst_rec16(2),0);
    Screen('DrawText',video.h,inst_txt17,inst_rec17(1),inst_rec17(2),0);
    Screen('DrawText',video.h,inst_txt18,inst_rec18(1),inst_rec18(2),0);
end

function load_pre_prac
    prac_txt1 = 'Now, we are going to do some training rounds.';
    prac_txt2 = 'The goal of this training is to understand';
    prac_txt3 = 'how the game works and see some of the quirks';
    prac_txt4 = 'you''ll encounter throughout the game.';
    prac_txt5 = 'For example, you''ll notice that the letters are not';
    prac_txt6 = 'always in the same position after each trial.';
    prac_txt7 = 'So pay constant attention to where your desired letter is';
    prac_txt8 = 'to not accidentally lose points!';

    prac_rec1 = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt1),video.x/2,1.0*ppd);
    prac_rec2 = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt2),video.x/2,3.0*ppd);
    prac_rec3 = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt3),video.x/2,4.5*ppd);
    prac_rec4 = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt4),video.x/2,6.0*ppd);
    prac_rec5 = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt5),video.x/2,8.0*ppd);
    prac_rec6 = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt6),video.x/2,9.5*ppd);
    prac_rec7 = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt7),video.x/2,11.5*ppd);
    prac_rec8 = CenterRectOnPoint(Screen('TextBounds',video.h,prac_txt8),video.x/2,13.0*ppd);

    Screen('DrawText',video.h,prac_txt1,prac_rec1(1),prac_rec1(2),0);
    Screen('DrawText',video.h,prac_txt2,prac_rec2(1),prac_rec2(2),0);
    Screen('DrawText',video.h,prac_txt3,prac_rec3(1),prac_rec3(2),0);
    Screen('DrawText',video.h,prac_txt4,prac_rec4(1),prac_rec4(2),0);
    Screen('DrawText',video.h,prac_txt5,prac_rec5(1),prac_rec5(2),0);
    Screen('DrawText',video.h,prac_txt6,prac_rec6(1),prac_rec6(2),0);
    Screen('DrawText',video.h,prac_txt7,prac_rec7(1),prac_rec7(2),0);
    Screen('DrawText',video.h,prac_txt8,prac_rec8(1),prac_rec8(2),0);
end

end % main function

% local functions
function [techn_order] = set_techn(mod6)
    techn_order = perms([1 2 3]); % the order of the result of the function perms is deterministic
    techn_order = techn_order(mod6,:);
end