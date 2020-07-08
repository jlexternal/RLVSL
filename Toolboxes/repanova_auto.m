function repanova_auto(x,s)

xsiz = size(x);

x = reshape(x,xsiz(1),[]);

nfac = xsiz(end:-1:2);
sfac = s(end:-1:1);

repanova(x,nfac,sfac);

end