function B_out = mrfRegularization(B,Q,opts)

siz = opts.size;
N   = prod(siz);
G   = opts.Transition;
C   = size(B,2);
rad = 1;
B_out = B;
parfor r = 1:N  
  [i1,i2,i3] = ind2sub(siz,r);
  l1  = max(min([-rad rad]+i1,siz(1)),1);
  l2  = max(min([-rad rad]+i2,siz(2)),1);
  l3  = max(min([-rad rad]+i3,siz(3)),1);
  [i1,i2,i3] = ndgrid(l1(1):l1(2),...
                      l2(1):l2(2),...
                      l3(1):l3(2));
  ind = sub2ind(siz,i1(:),i2(:),i3(:));  
  pr  = Q(setdiff(ind,r),:);
%   pr   = pr./repmat(sum(pr,1),[size(pr,1) 1]);
  B_out(r,:)= B(r,:).*exp(-sum(pr,1)*G);
end
B1  = sum(B_out,2);
parfor c = 1:C
  B_out(:,c) = B(:,c)./B1;
end