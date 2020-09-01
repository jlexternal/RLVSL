function fn_hello_world(iter)
    out = struct;
    out.iter = iter;
    savename = ['hello_world_iter_' sprintf('%02d',num2str(iter)) '_' datestr(now,'ddmmyyyy')];
    save(savename,'out');
end