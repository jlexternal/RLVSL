function parse_eyelink(filename)
%  PARSE_EYELINK  Parse EyeLink data in ASCII format
%
%  Usage: PARSE_EYELINK(filename)
%
%  where filename - EyeLink data filename in ASCII format
%
%  The function writes a MAT file with the same name as the input file and into
%  the same folder.
%
%  Valentin Wyart <valentin.wyart@ens.fr>

if nargin < 1
    error('Missing input filename!');
end

% check input filename extension
[~,~,fext] = fileparts(filename);
if ~strcmp(fext,'.asc')
    error('Input filename should have .asc extension!');
end

% open file handle
fid = fopen(filename,'rt');

% define data information
hdr  = {}; % data header
fsmp = []; % sampling frequency (Hz)
tini = []; % initialization time
tsmp = []; % time of eye sample
xsmp = []; % x-position of eye sample
ysmp = []; % y-position of eye sample
psmp = []; % pupil diameter of eye sample
tsac = []; % time of detected saccade
xsac = []; % x-positions of start/end of detected saccade
ysac = []; % y-positions of start/end of detected saccade
tbli = []; % time of detected blink
tmsg = []; % time of message
smsg = {}; % message string

% parse loop
fprintf('\n');
fprintf('parsing EyeLink data in ASCII format...\n');
nsmp = 0;
while true
    
    lin = fgetl(fid);
    if lin == -1
        break
    end
    
    c = strsplit(lin);

    if ~isempty(str2num(c{1})) % eye sample
        nsmp = nsmp+1;
        
        if nsmp > length(tsmp)
            fprintf('  => %4d seconds read so far',floor(nsmp/fsmp));
            if length(tsmp) > 0
                fprintf(' (proportion missing = %.1f %%)\n',mean(isnan(psmp))*100);
            else
                fprintf('\n');
            end
            tsmp = cat(1,tsmp,nan(30*fsmp,1));
            xsmp = cat(1,xsmp,nan(30*fsmp,1));
            ysmp = cat(1,ysmp,nan(30*fsmp,1));
            psmp = cat(1,psmp,nan(30*fsmp,1));
        end
        
        tsmp(nsmp,1) = str2num(c{1});
        if any(cellfun(@(s)strcmp(s,'.'),c(2:4))) % missing?
            xsmp(nsmp,1) = nan;
            ysmp(nsmp,1) = nan;
            psmp(nsmp,1) = nan;
        else
            xsmp(nsmp,1) = str2num(c{2});
            ysmp(nsmp,1) = str2num(c{3});
            psmp(nsmp,1) = str2num(c{4});
        end
        
    else % extra information
        
        % header information
        if strcmp(c{1},'**') && length(c) > 1
            hdr{end+1,1} = strjoin(c(2:end));
        end
        
        % initialization time
        if strcmp(c{1},'START')
            tini = str2num(c{2});
        end
        
        % trigger
        if strcmp(c{1},'MSG')
            if length(c) == 3 % trigger message
                tmsg(end+1,1) = str2num(c{2});
                smsg{end+1,1} = c{3};
            else % extra header information
                hdr{end+1,1} = strjoin(c(3:end));
                if strcmp(c{3},'!MODE')
                    fsmp = str2num(c{6});
                end
            end
        end
        
        % saccade
        if strcmp(c{1},'ESACC') && ~any(cellfun(@(s)strcmp(s,'.'),c(6:9)))
            tsac(end+1,:) = [str2num(c{3}),str2num(c{4})];
            xsac(end+1,:) = [str2num(c{6}),str2num(c{8})];
            ysac(end+1,:) = [str2num(c{7}),str2num(c{9})];
        end
        
        % blink
        if strcmp(c{1},'EBLINK')
            tbli(end+1,:) = [str2num(c{3}),str2num(c{4})];
        end
        
    end
    
end
fprintf('reached end-of-file.\n\n');

% close file handle
fclose(fid);

% save output to disk
[fpath,fname] = fileparts(filename);
filename = fullfile(fpath,[fname,'.mat']);
save(filename, ...
    'hdr','fsmp','tini','tsmp','xsmp','ysmp','psmp','tsac','xsac','ysac','tbli','tmsg','smsg');

end