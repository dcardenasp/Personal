%BY E-M

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

method = 1; %0: 1-Normal by class. 1: parzen full. 2: parzen block. 3: parzen local. 4: Parzen coord as feature
regularization = false;
masked = true;
V = spm_vol('/media/dcardenasp/VERBATIM/DataBases/subject_01/T1_1.nii');
K  = 1000;
M = 5e3;
B = 3;
radius = 20;
maxiter = 20;
converg = 1e-4;
alpha = 0.3;
beta  = 5;
step = 0.1;

X = spm_read_vols(V);
siz = size(X);
N = prod(siz);
list = [6 5 4 3 1 2];
numClasses = numel(list);
Pr0 = zeros([siz numClasses]);
parfor l=1:numel(list)
  Pr0(:,:,:,l) = spm_read_vols(spm_vol(...
    ['/media/dcardenasp/VERBATIM/DataBases/subject_01/c' num2str(list(l)) '_1.nii']));
end

Pr0  = Pr0 - min(Pr0(:));
sPr = sum(Pr0,4);
Pr0  = Pr0./repmat(sPr,[1 1 1 size(Pr0,4)]);
Pr0  = reshape(Pr0,prod(siz),size(Pr0,4));

G = transMatrix(numClasses,alpha,beta);

X = reshape(X,[N 1]);

if method==1
  Ind = randperm(numel(X),K);
%   w = ones(K,numClasses);  
  x = X(Ind)';
  X = reshape(X,[N 1]);
  w = Pr0(Ind,:);  
  w = w./repmat(sum(w,1),[K 1]);
end

opts.M = M;
opts.K = K;
opts.indices = Ind;
% opts.sigma = s;
opts.MaxIter = maxiter;
opts.minEnergyDiff = converg;
opts.size = siz;
opts.radius = radius;
opts.Transition = G;
opts.mrf = regularization;
opts.numClasses = numClasses;
switch method
case 0
  opts.mixture = ones(1,numClasses)/numClasses;
  opts.covariance = eye(4);%[s^2 0 0 0; 0 5^2 0 0; 0 0 5^2 0; 0 0 0 5^2];
case 1
  opts.mixture = w;
  opts.samples = reshape(X(Ind),K,1);
  opts.covariance = 50^2;
end
opts.step = step;

err  = zeros(1,opts.MaxIter);
ener = zeros(1,opts.MaxIter);

Pos = Pr0;
[~,L1] = max(Pos,[],2);
L1  = (reshape(L1,siz)-1)>0;
SE  = strel(ones(20,20,20));
L2  = imdilate(1.0*L1,SE);
ind = L2(:)>0;

[R1,R2,R3] = ndgrid(1:siz(1), 1:siz(2), 1:siz(3));

% Ind = round(linspace(1,N,K));
% Y = [X(ind,:) R1(ind') R2(ind') R3(ind')];
Y = X(ind,:);
Y = zscore(Y);
N = size(Y,1);
Ind = randperm(N,K);
y = Y(Ind,:);

for iter=1:opts.MaxIter
    W = Pos(ind,:);
    N = size(Y,1);
    w = W(Ind,:);
    %w = w./repmat(sum(w,1),[K 1]);
    
    if regularization
      Pr = mrfRegularization(Pr0,Pos,opts);
    else
      Pr = Pr0;
    end

    clear Cov
    for c=1:numClasses
      Cov(:,:,c) = weightedcov(Y,W(:,c));
    end

    F = zeros(N,numClasses);
    parfor r=1:N
      for c=1:numClasses
        F(r,c) = mvnpdf(y,Y(r,:),Cov(:,:,c))'*w(:,c);
      end
    end
    F2 = zeros([prod(siz) numClasses]);
    F2(:,1)=1.0;
    F2(ind,:) = F;
    
    Pos = F2.*Pr;
    Pos = Pos./repmat(sum(Pos,2),[1 numClasses]);    
    
    ener(iter) = BayesEnergy(Pr,F2);
    figure(1)
    plot(1:iter,ener(1:iter))
    drawnow
end

[~,L1] = max(Pr0,[],2);
L1 = reshape(L1,siz)-1;
[~,L2] = max(Pos,[],2);
L2 = reshape(L2,siz)-1;
V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/seg_Pr.nii';
spm_write_vol(V,L1);
V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/seg_PosParzenReg.nii';
spm_write_vol(V,L2);
% V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/Pc_1.nii';
% spm_write_vol(V,L3);