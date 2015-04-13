clear all
close all
clc

if ismac
  path_base = '/Users/dcardenasp/Documents/MATLAB';
elseif isunix
  path_base = '/home/dcardenasp/Documents/MATLAB';
end
addpath(genpath(fullfile(path_base,'SliceBrowser')))
addpath(genpath(fullfile(path_base,'spm8')))
addpath(genpath(fullfile(path_base,'MLmat')))

M = 2000;

X = spm_read_vols(spm_vol('/media/dcardenasp/VERBATIM/MRI/OAS1_0001_MR1/mri/T1.nii'));
siz = size(X);
N = prod(siz);
Pr = zeros([siz 6]);
list = [6 5 4 3 1 2];
parfor l=1:numel(list)
  Pr(:,:,:,l) = spm_read_vols(spm_vol(...
    ['/media/dcardenasp/VERBATIM/MRI/OAS1_0001_MR1/tissue/c' num2str(list(l)) '.nii']));
end
X   = reshape(X,[N 1]);
B   = reshape(Pr,[N 6]);

%% Augment input space:
tmp = zeros(N,27);
for i=1:27
  mask = zeros(3,3,3);
  mask(i) = 1;
  Y = convn(reshape(X,siz),mask,'same');
  tmp(:,i) = Y(:);
end
X0 = X;
[X1,X2,X3] =  ndgrid(1:siz(1),1:siz(2),1:siz(3));
X  = [tmp X1(:) X2(:) X3(:)];
X  = zscore(X);

clear tmp Y X1 X2 X3
%%

[~,L] = max(Pr,[],4);
Fg = L>1;
se = strel(ones(11,11,11));
Fg = imdilate(Fg,se);
ind1 = Fg(:)>0;
X1   = X(ind1,:);
B1   = B(ind1,:);
N    = sum(ind1);
ind2 = randperm(N,M);
X1  = X1(ind2,:);
B1  = B1(ind2,:);

L1 = L(ind1);
L1 = L1(ind2);
% [~,ind3] = sortrows(B1,[1 2 3 4 5 6]);
[~,ind3] = sort(L1);
X1 = X1(ind3,:);
B1 = B1(ind3,:);

Dx  = pdist(X1);
Db  = pdist(B1);

sb = kScaleOptimization(B1);
Kb = kExpQuad(squareform(Db),sb,'distances');
[Ax,sx] = kMetricLearningMahalanobis(X1,Kb,[],size(X1,2),true,[5e-4 5e-4]);
y=X1*Ax;
d=pdist2(y,y);
Ky=exp(-d.^2/2);
%%

Dx = squareform(Dx);
Db = squareform(Db);

figure(3)
subplot(2,2,1)
imagesc(Dx)
axis square
subplot(2,2,2)
imagesc(Db)
axis square
subplot(2,2,3)
imagesc(Ky,[0 1])
axis square
subplot(2,2,4)
imagesc(Kb,[0 1])
axis square