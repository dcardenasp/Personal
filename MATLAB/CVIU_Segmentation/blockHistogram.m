clear all;
close all;
clc;

clear all
close all
clc

addpath(genpath('/home/dcardenasp/Documents/MATLAB/SliceBrowser'))
addpath(genpath('/home/dcardenasp/Documents/MATLAB/spm8'))
addpath(genpath('/home/dcardenasp/Documents/MATLAB/MLmat'))

V = spm_vol('T1_1.nii');
K  = 1000;
M = 2e4;
B = 3;

X = spm_read_vols(V);
Pr = spm_read_vols(spm_vol('white_1.nii'));
Pr(:,:,:,2) = spm_read_vols(spm_vol('grey_1.nii'));
Pr(:,:,:,3) = spm_read_vols(spm_vol('csf_1.nii'));
sPr = sum(Pr(:,:,:,1:3),4);
Pr = reshape(Pr,[numel(X) 3]);
Pr(sPr(:)>1.0,:) = Pr(sPr(:)>1.0,:) - repmat((sPr(sPr(:)>1.0)-1)/3,[1 3]);
Pr(:,4) = 1-sum(Pr(:,1:3),2);
siz = size(X);
N = numel(X);

ind  = randperm(floor(N/(B^3)),K);
blkSiz = round(siz/B);

y = (0:600)';

i=1;
for b1=0:B-1;
for b2=0:B-1;
for b3=0:B-1;
  
%   if any([b1 b2 b3]>0)
    b = sub2ind(siz, b1*blkSiz(1)+1, b2*blkSiz(2)+1, b3*blkSiz(3)+1);
%   else
%     b = 0;
%   end
  ind2 = ind+b;
    
  x = reshape(X(ind2),[K 1]);
  d = pdist(x);
  s = kScaleOptimization(d);
  A = kExpQuad2(y,x,s);
  
  pr = Pr(ind2,:);
  w = pr./repmat(sum(pr),[K 1]);
  w(isnan(w)) = 0;
  
  P = A*w;
  subplot(ceil(sqrt(B^3)),ceil(sqrt(B^3)),i)
  plot(y,P)
  i=i+1;  
end
end
end

%%
figure(2)
for i=1:B^3
  ind2=Ind{i};
  x = reshape(X(ind2),[K 1]);
  d = pdist(x);
  s = kScaleOptimization(d);
  A = kExpQuad2(y,x,s);
  
  pr = Pr(ind2,:);
  w = pr./repmat(sum(pr),[K 1]);
  w(isnan(w)) = 0;
  
  P = A*w;
  subplot(ceil(sqrt(B^3)),ceil(sqrt(B^3)),i)
  plot(y,P)
end