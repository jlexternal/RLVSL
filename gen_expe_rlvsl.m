function [expe] = gen_expe_rlvsl(subj)
%
%  Usage: [expe] = GEN_EXPE_RLVSL(subj)
%
%  where subj is the subject number
%        expe is the generated experiment structure
%
%  The experiment structure expe contains the following fields:  
%   * type      - block type     ('training', or test conditions:
%                                 'random', 'repeating', 'alternating')
%   * sesh      - session number (0:training)
%   * blck      - blocks with value differences given correct choice
%   * shape     - bandit shapes
%   * color     - colors for each of the 4 conditions
%   * locat     - correct bandit location (1:left, 2:right)
%
%  This function generates an experimental structure based on the seed indicated via
%  the subject number. 
%   
%  Requires: gen_blck_rlvsl(cfg)
%
%  Note: One pair of shapes is associated with one condition color throughout the
%         entire experiment
% 
% Jun Seok Lee - Oct 2019

% Check input argument
if nargin < 1
    error('Missing subject number!');
end

% Add toolboxes to path
addpath('./Toolboxes/Rand');

% Initialize random number generator
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

expe = [];

% Set parameters for TRAINING blocks (default)
cfg_trn         = struct;
cfg_trn.ntrls   = 16;   % maybe increase this during training? to 15 or so?
cfg_trn.mgen    = .1;
cfg_trn.sgen    = .2;
cfg_trn.nbout   = 3;    % 3 training blocks
cfg_trn.ngen    = 10000;

% Set parameters for REAL blocks (default)
nblckpersesh    = 6; % should be a multiple of condpersesh
nsessions       = 8; % number of sessions
condpersesh     = 3; % conditions per session
cfg = struct;
cfg.ntrls   = 16;
cfg.mgen    = .1;
cfg.sgen    = .2;
cfg.nbout   = nblckpersesh*nsessions; % 9 blocks x 8 sessions = 72 total blocks
cfg.ngen    = 10000;
cfg.mcrit   = .05;      % max distance of sampling mean from true mean
cfg.scrit   = .1;      % max difference between sampling spread and true spread
cfg.ntrain  = cfg_trn.nbout;

expe(1).cfg = cfg;

% Establish bandit shape-correctness associations
% ("Which shape corresponds to correct choice?")
%   shapes -  1:circle, 2:triangle, 3:diamond, 4:pentagon, 5:star, 6:plus
%   (training 7:A,       8:B )
%   correct shape in each condition determined by index (1:correct, 2:not)
shape_list      = randperm(6);
shape_list_trn  = [7 8; 9 10; 11 12]; % randomization happens below

% Establish (background) color-condition associations
% ("Which color corresponds to which pattern?")
%   color         - 1:red,    2:blue,     3:grey
%   cond index    - 1:rep,    2:alt,      3:rand 
repaltorder = perms([1 2]);
color_list = [repaltorder(mod(subj,2)+1,:) 3];

% Establish episode layout 
% ("Where is the correct bandit on each trial?")
%   1:left, 2:right
correct_loc_trn = randi([1 2], [cfg_trn.nbout cfg_trn.ntrls]);
correct_loc     = randi([1 2], [cfg.nbout cfg.ntrls]);

% Establish order of conditions per session
sesh_order = rand(nsessions,condpersesh);
[~,sesh_order] = sort(sesh_order,2);
sesh_order = [sesh_order sesh_order sesh_order];

% Generate training blocks
%   maybe use a set of shapes that are not used at all in the real blocks
%   mainly to explain that the goal of the task is to ALWAYS choose 
%       best option (regardless of evidence)

blck = gen_blck_rlvsl(cfg_trn);
ntbs = cfg_trn.nbout;
for i = 1:ntbs
    expe(i).sesh  = 0; 
    expe(i).type = 'training';
    expe(i).blck = blck(i,:);
    expe(i).shape = shape_list_trn(i,randperm(2)); % corresponding to A or B
    expe(i).color = 4; % 4:grey, since training block
    expe(i).locat = correct_loc_trn(i,:);
end

% Generate true testing blocks
blck = gen_blck_rlvsl(cfg);

% Choose the highest ranked subset of these blocks to be used in ALL 3 conditions
nblckpercond = nblckpersesh/condpersesh*nsessions;
blck = blck(1:nblckpercond,:); % 16 chosen blocks
blck_index = [randperm(nblckpercond); randperm(nblckpercond); randperm(nblckpercond)];

iblock = 1;
ib_c  = ones(3,1);
is_alt_flip = false;
rnd_sesh_seq = [];
rnd_sesh_ibs = [];
for isesh = 1:nsessions

    sesh_cond = ones(1,nsessions);
    for jblck = 1:nblckpersesh
        sesh_cond(jblck) = sesh_order(isesh,jblck);
    end
    
    iblock_local = 1;
    endsesh = false;
    while ~endsesh
        condind = sesh_cond(iblock_local);
        switch condind 
            case 1
                condtype    = 'rep';
                shapes      = shape_list(1:2);
                colors      = color_list(1);
                ic          = 1;
            case 2 
                condtype    = 'alt';
                shapes      = shape_list(3:4);
                colors      = color_list(2);
                ic          = 2;
            case 3
                condtype    = 'rnd';
                shapes      = shape_list(5:6);
                colors      = color_list(3);
                ic          = 3;
        end
        expe(iblock+ntbs).sesh  = isesh;
        expe(iblock+ntbs).type  = condtype;
        if strcmp(condtype,'alt') % special condition: shape_list will reverse order 
            if is_alt_flip
                expe(iblock+ntbs).shape = flip(shapes);
                is_alt_flip = false;
            else
                expe(iblock+ntbs).shape = shapes;
                is_alt_flip = true;
            end
        elseif strcmp(condtype,'rnd') 
            expe(iblock+ntbs).shape = datasample(shapes,2,'Replace',false);
            rnd_sesh_seq = [rnd_sesh_seq expe(iblock+ntbs).shape(1)];
            rnd_sesh_ibs = [rnd_sesh_ibs iblock+ntbs];
        else
            expe(iblock+ntbs).shape = shapes;
        end
        
        % Assign outcomes for block
        expe(iblock+ntbs).blck  = blck(blck_index(ib_c(ic)),:);
        
        % Assign color for the block
        expe(iblock+ntbs).color = colors; 
        expe(iblock+ntbs).locat = correct_loc(iblock,:);
    
        if mod(iblock,nblckpersesh) == 0 
            iblock          = iblock + 1;
            iblock_local    = iblock_local + 1;
            endsesh = true; % indicate end of session
        else
            iblock          = iblock + 1;
            iblock_local    = iblock_local + 1;
        end
        ib_c(ic) = ib_c(ic) + 1;
    end
    
    % The comment below pertains to if the 'rnd' condition is tracked with the Beta
    % distribution. Otherwise, this step profers no benefit.
    %   Check for pseudorandomness in the 'rnd' condition at the end of a quarter
    %   (i.e. ensure that no session of 'rnd' shows the same patterns of 'rep' or 'alt')
    if mod(isesh,2) == 0
        nblckpercond = nblckpersesh/condpersesh;
        rnd_sesh_seq(rnd_sesh_seq == rnd_sesh_seq(1)) = 1;
        rnd_sesh_seq(rnd_sesh_seq ~= 1) = 0;
        rnd_flip_blk = datasample(rnd_sesh_ibs,1);
        
        if isequal(rnd_sesh_seq,ones(1,nblckpercond*2)) | isequal(rnd_sesh_seq,zeros(1,nblckpercond*2)) % if repeating
            expe(rnd_flip_blk).shape = flip(expe(rnd_flip_blk).shape);
        elseif isequal(rnd_sesh_seq,repmat([1 0],1,nblckpercond)) | isequal(rnd_sesh_seq,repmat([0 1],1,nblckpercond)) % if alternating
            expe(rnd_flip_blk).shape = flip(expe(rnd_flip_blk).shape);
        end
        
        rnd_sesh_seq = []; % reset for next quarter
        rnd_sesh_ibs = []; % reset for next quarter
    end
      
end
