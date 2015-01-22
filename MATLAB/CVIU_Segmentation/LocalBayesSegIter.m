function [Pos, Pc, w, sigma] = LocalBayesSegIter(X,Pr,opts)

N   = numel(X);
s   = opts.sigma;
siz = opts.size;
rad = opts.radius;
K   = opts.K;

Pc = zeros([numel(X) size(Pr,2)]);
num = zeros(1,N);
den = num;
parfor r = 1:N  
  [i1,i2,i3] = ind2sub(siz,r);
  i1  = max(min([-rad rad]+i1,siz(1)),1);
  i2  = max(min([-rad rad]+i2,siz(2)),1);
  i3  = max(min([-rad rad]+i3,siz(3)),1);
  [i1,i2,i3] = ndgrid(i1(1):i1(2),i2(1):i2(2),i3(1):i3(2));
  ind = sub2ind(siz,i1(:),i2(:),i3(:));
  p   = randperm(numel(ind),min(numel(ind),K));
  ind = ind(p);
  pr  = Pr(ind,:);
  pr   = pr./repmat(sum(pr),[numel(ind) 1]);
  x   = reshape(X(ind),[numel(ind) 1]);  
  Pc(r,:) = kExpQuad2(X(r),x,s)*pr;
   
  num(r) = ((X(r)-x').^2)*pr*Pr(r,:)';
  den(r) = sum(pr*Pr(r,:)');
end
w = Inf;
sigma = sqrt(sum(num)/sum(den));
Pos = Pc.*Pr;
Pos = Pos./repmat(sum(Pos,2),[1 size(Pc,2)]);  