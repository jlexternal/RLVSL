function [ind,imsg] = pupil_event_indexer_rlvsl(data_eye)
% pupil_event_indexer_rlvsl
%
% Usage: Indexes the location of events of interest in the preprocessed data structure in
% experiment RLVSL.
%
% Input:  data_eye - preprocessed pupil data structure for a single block of a single subject 
%
% Output: ind      - logical array indicating locations of events corresponding to the time
%                    sampling array
%         imsg     - numerical index of the event in the sampling array
%
% Jun Seok Lee <jlexternal@gmail.com>

tsmp = data_eye.tsmp;
tmsg = data_eye.tmsg;
smsg = data_eye.smsg;

n_unique_events = 4; % the number of different types of events (e.g. STIM,RESP,FBCK,END)

len_tsmp = length(tsmp);

imsg = zeros(length(smsg),1);
ind  = zeros(len_tsmp,4);

ptr_smsg = 1;
ind_bool = false;
for i = 1:len_tsmp-1
    if tmsg(ptr_smsg) > tsmp(i) && tmsg(ptr_smsg) < tsmp(i+1)
        % if the event happens in between two sampling points, 
        % attribute to the latter point
        j = i+1;
        ind_bool = true;
    elseif tmsg(ptr_smsg) == tsmp(i)
        % if the event happens on the sampling point,
        % attribute to that point
        j = i;
        ind_bool = true;
    elseif tmsg(ptr_smsg) == tsmp(i) && ptr_smsg > 1
        % if two consecutive events fall on the same time sample
        j = i-1;
        ind_bool = true;
    end
    
    if ind_bool
        event = mod(ptr_smsg,n_unique_events); % recall: n_unique_events = 4
        switch event
            case 1 % STIM
                ind(j,1) = 1;
            case 2 % RESP
                ind(j,2) = 1;
            case 3 % FBCK
                ind(j,3) = 1;
            case 0 % END
                ind(j,4) = 1;
        end
        imsg(ptr_smsg) = j;      % log index of the event in sampling array
        ptr_smsg = ptr_smsg + 1; % move the pointer along smsg
        
        if ptr_smsg<length(tmsg) && tmsg(ptr_smsg) == tmsg(ptr_smsg-1) % if two events occur concurrently
            imsg(ptr_smsg) = j;
            ptr_smsg = ptr_smsg + 1;
        end
        ind_bool = false;
        
        if ptr_smsg > length(smsg)
            break;
        end
    end
end

end