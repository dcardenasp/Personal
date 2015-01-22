function [Pos, Pc, w] = BayesSegIter(Y,q,opts)

N = numel(Y);
s   = opts.sigma;
ind = opts.indices;
K   = opts.K;
M   = opts.M;
x   = Y(ind)';

pr = q(ind,:);
w = pr./repmat(sum(pr),[K 1]);
Pc = zeros([size(Y) size(q,2)]);
parfor r = 1:ceil(N/M)
  Pc(:,r,:) = kExpQuad2(Y(:,r),x,s)*w;  
end

Pc=reshape(Pc,size(Pc,1)*size(Pc,2),size(Pc,3));
Pc = Pc(1:size(q,1),:);
Pos = Pc.*q;
Pos = Pos./repmat(sum(Pos,2),[1 size(Pc,2)]);  