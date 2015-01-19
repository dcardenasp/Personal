function [s_opt,fval,exitflag,output,grad] = bayesScaleOptimization(X,Y,W,Pr,s0)

if nargin < 5
  s0 = median(pdist(Y));
end

opts = optimoptions('fminunc','GradObj','on','Display','iter');
[s_opt,fval,exitflag,output,grad] = fminunc(@(s)myfun(s,X,Y,W,Pr),s0,opts);

function [f,df] = myfun(s,X,Y,W,Pr)

tic
f  = zeros(size(X));
df = zeros(size(X));

parfor r = 1:numel(X);
  K     = normpdf(Y,X(r),s);
  den   = K'*W*Pr(r,:)';
  f(r)  = log(den);  
  D     = (X(r)-Y).^2;
  num   = s*(D.*K)'*W*Pr(r,:)';
  df(r) = num/den;
end
f  = -sum(f);
df = -sum(df);
toc