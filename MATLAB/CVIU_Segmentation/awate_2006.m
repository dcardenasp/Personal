clear all
close all
load awate_2006.mat

siz     = size(X);
N       = prod(siz);
X       = reshape(X,[N 1]);
L       = reshape(L,[N 1]);
s_spat  = 15;
K       = 200;
Neighs  = round(s_spat*randn(K,numel(siz)));
Classes = 0:5;
F       = zeros(N,numel(Classes));

for r=1:N
  [r1,r2,r3]=ind2sub(siz,r);
  rNeighs = Neighs + repmat([r1 r2 r3],[K 1]);
  if all(rNeighs(:,1)<siz(1)) && all(rNeighs(:,2)<siz(2)) && all(rNeighs(:,3)<siz(3))...
     && all(rNeighs(:,1)>0) && all(rNeighs(:,2)>0) && all(rNeighs(:,3)>0)
     
    inds = sub2ind(siz,rNeighs(:,1),rNeighs(:,2),rNeighs(:,3));
    x    = X(inds,:);
    l    = L(inds);
    for c=1:numel(Classes)
      x2 = x(l==Classes(c));
      F(r,c) = mean(mvnpdf(x2,X(r),80));
    end
  end
  if mod(r,10000)==0
    fprintf('%d - %d (%2.2f)\n',r,N,r/N*100)
  end
end