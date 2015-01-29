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
regularization = true;
masked = true;
V = spm_vol('/media/dcardenasp/VERBATIM/DataBases/subject_01/T1_1.nii');
K  = 100;
M = 5e3;
B = 3;
radius = 20;
maxiter = 20;
converg = 1e-4;
alpha = 0.3;
beta  = 5;
step = 0.1;

y = (0:600)';

X = spm_read_vols(V);
siz = size(X);
N = prod(siz);
list = [6 5 4 3 1 2];
numClasses = numel(list);
Pr = zeros([siz numClasses]);
parfor l=1:numel(list)
  Pr(:,:,:,l) = spm_read_vols(spm_vol(...
    ['/media/dcardenasp/VERBATIM/DataBases/subject_01/c' num2str(list(l)) '_1.nii']));
end

Pr  = Pr - min(Pr(:));
sPr = sum(Pr,4);
Pr  = Pr./repmat(sPr,[1 1 1 size(Pr,4)]);
Pr  = reshape(Pr,prod(siz),size(Pr,4));

G = transMatrix(numClasses,alpha,beta);


if method==1
  Ind = randperm(numel(X),K);
  w = ones(K,numClasses);  
%   x = X(Ind)';
%   X = reshape(X,[N 1]);
%   w = Pr(Ind,:);  
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
  opts.covariance = 100^2;
end
opts.step = step;

err  = zeros(1,opts.MaxIter);
ener = zeros(1,opts.MaxIter);

Pos = Pr;
for iter = 1:opts.MaxIter
  ticid=tic;
  
  if masked
    [~,L1] = max(Pos,[],2);
    L1  = (reshape(L1,siz)-1)>0;
    SE  = strel('ball',10,10,8);
    L2  = imdilate(1.0*L1,SE);
    ind = L2(:)>10;
    ind = randperm(N,5*K);
  else
    ind = true(N,1);
  end
  
  Pos_ant = Pos;
%   w_ant = w;
  switch method
  case 0
    opts = BayesNormalMStep(X(ind),Pr(ind,:),Pos(ind,:),opts); %M-step
    [Pos,ener(iter)] = BayesNormalEStep(X,Pr,opts); %E-step   
  case 1
    F = BayesLikelihood(X(:),opts);
    [opts.w, opts.covariance] = BayesParzenMStep(X(ind)',F(ind,:),Pos(ind,:),opts); %M-step
  end
  
  t = toc(ticid);
  
  ener(iter) = BayesEnergy(Pr,F);
  figure(1)
  plot(1:iter,ener(1:iter))
  drawnow  
  
%   ed = Inf;
%   if iter>5    
%     ed = abs(ener(iter)-ener(iter-1));
%     if ed<opts.minEnergyDiff
%       break;
%     end
%   end  
  fprintf('Iteration %d/%d. ED %.2f. WD %.2e. Time %.2f\n',iter,opts.MaxIter,Inf,Inf,t)  
end

[~,L1] = max(Pr,[],2);
L1 = reshape(L1,siz)-1;
[~,L2] = max(Pos,[],2);
L2 = reshape(L2,siz)-1;
% [~,L3] = max(Pc,[],2);
% L3 = reshape(L3,siz)-1;
V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/segPr.nii';
spm_write_vol(V,L1);
V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/segPosNorm.nii';
spm_write_vol(V,L2);
% V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/Pc_1.nii';
% spm_write_vol(V,L3);