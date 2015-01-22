function [Pos, Pc, w] = BayesSegSpaceIter(X,Pr,opts)

s   = opts.sigma;
cov = opts.covariance;
siz = opts.size;  
ind = opts.indices;
N   = prod(siz);
K   = opts.K;
M   = opts.M;
x   = reshape(X(ind),[numel(ind) 1]);
pos = zeros(numel(ind),3);
[pos(:,1),pos(:,2),pos(:,3)] = ind2sub(siz,ind);
x   = [x pos];

pr = Pr(ind,:);
w = pr./repmat(sum(pr),[K 1]);
Pc = zeros([N size(Pr,2)]);
parfor r = 1:N
  pos = zeros(1,3);
  [pos(1),pos(2),pos(3)] = ind2sub(siz,r);
  Pc(r,:) = mvnpdf(x,[X(r) pos],cov)'*w;
end
Pos = Pc.*Pr;
Pos = Pos./repmat(sum(Pos,2),[1 size(Pc,2)]);  