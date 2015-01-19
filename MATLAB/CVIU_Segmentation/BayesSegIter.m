function [Pos, Pc, w] = BayesSegIter(X,Pr,opts)

N = numel(X);

if nargin<3
  K=1000;
  M = 1e3;
  
  ind  = randperm(numel(X),K);
  x = X(ind)';
  X = X(:);
  
  d = pdist(x);
  s = kScaleOptimization(d);
  
  X = [X;ones(ceil(N/M)*M-N,1)];
  X = reshape(X,M,ceil(N/M));
else
  s   = opts.sigma;
  ind = opts.indices;
  K   = opts.K;
  M   = opts.M;
  x   = X(ind)';
end

pr = Pr(ind,:);
w = pr./repmat(sum(pr),[K 1]);
Pc = zeros([size(X) size(Pr,2)]);
parfor r = 1:ceil(N/M)
  Pc(:,r,:) = kExpQuad2(X(:,r),x,s)*w;  
end

Pc=reshape(Pc,size(Pc,1)*size(Pc,2),size(Pc,3));
Pc = Pc(1:size(Pr,1),:);
Pos = Pc.*Pr;
Pos = Pos./repmat(sum(Pos,2),[1 size(Pc,2)]);  