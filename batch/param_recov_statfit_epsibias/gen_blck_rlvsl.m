function [blck] = gen_blck_rlvsl(cfg)
%
%  Usage: [blck] = GEN_BLCK_RLVSL(cfg)
%
%  where cfg is a configuration structure
%        blck is the generated block structure
%
%  The configuration structure cfg contains the following fields:
%   * ntrls     - number of trials per block
%   * mgen      - mean of generative distribution
%   * sgen      - standard devation of generative distribution
%   * ngen      - number of blocks to be selected from
%   * nbout     - number of blocks to output
%
%   Potentially change these below:
%    * m_crit    - max distance of sampling mean from true mean
%    * s_crit    - max difference of sampling spread from true spread
%
% This function samples from a true generative distribution, and outputs a set of
% samples given some criteria. 
%
% Note: There is no consideration of the condition structure (e.g. repeating,
%       alternating, or random). It simply outputs the most "average" sequence of
%       tracking values based on its dynamics over block length time.
% 
% Jun Seok Lee - Oct 2019

% check configuration structure
if ~all(isfield(cfg,{'ntrls','mgen','sgen','nbout'}))
    error('Incomplete configuration structure!');
end
if ~isfield(cfg,'ngen')
    cfg.ngen = 1e4;
end
if ~isfield(cfg,'mcrit')
    cfg.mcrit = .1;
end
if ~isfield(cfg,'scrit')
    cfg.scrit = .1;
end

% Add toolboxes to path
addpath('./Toolboxes/Rand');

% Initialize random number generator
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

% Localize configuration parameters
ntrls   = cfg.ntrls;
mgen    = cfg.mgen;
sgen    = cfg.sgen;
ngen    = cfg.ngen;
nbout   = cfg.nbout;
mcrit  = cfg.mcrit;
scrit  = cfg.scrit;

blck = [];

blocks = normrnd(mgen, sgen, [ngen ntrls]); % sample from distribution

% find means of blocks
m_blocks = mean(blocks,2);

% calculate spread of blocks
v_blocks = var(blocks,0,2);

% calculate distance measures for each block
m_blocks = abs(m_blocks - mgen);
v_blocks = abs(v_blocks - sgen^2);

% keep blocks within acceptable criteria
ind_crit = intersect(find(m_blocks<=mcrit),find(v_blocks<=scrit^2));
blocks = blocks(ind_crit,:);
m_blocks = m_blocks(ind_crit);
v_blocks = v_blocks(ind_crit);

% rank them
[~,imeans]      = sort(m_blocks,'ascend'); 
[~,ivars]       = sort(v_blocks,'ascend');
m_rank          = 1:length(m_blocks);
v_rank          = 1:length(v_blocks);
m_rank(imeans)  = m_rank;
v_rank(ivars)   = v_rank;
[~,iblocks]     = sort(m_rank+v_rank,'ascend');
iblocks         = iblocks(1:nbout);
blocks          = blocks(iblocks,:);

blck = blocks;



