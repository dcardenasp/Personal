clear all
close all
clc

addpath(genpath('/home/dcardenasp/Documents/MATLAB/SliceBrowser'))
addpath(genpath('/home/dcardenasp/Documents/MATLAB/spm8'))
addpath(genpath('/home/dcardenasp/Documents/MATLAB/MLmat'))
%addpath(genpath('/home/dcardenasp/Documents/MATLAB/fieldtrip'))


V = spm_vol('T1_1.nii');
K  = 1000;

X = spm_read_vols(V);
Pr = spm_read_vols(spm_vol('white_1.nii'));
Pr(:,:,:,2) = spm_read_vols(spm_vol('grey_1.nii'));
Pr(:,:,:,3) = spm_read_vols(spm_vol('csf_1.nii'));
sPr = sum(Pr(:,:,:,1:3),4);
Pr = reshape(Pr,[numel(X) 3]);
Pr(sPr(:)>1.0,:) = Pr(sPr(:)>1.0,:) - repmat((sPr(sPr(:)>1.0)-1)/3,[1 3]);
Pr(:,4) = 1-sum(Pr(:,1:3),2);
siz = size(X);
N = prod(siz);
M = 2e4;

ind  = randperm(numel(X),K);
x = X(ind)';
X = X(:);

d = pdist(x);
s = kScaleOptimization(d);

Y = [X;ones(ceil(N/M)*M-N,1)];
Y = reshape(Y,M,ceil(N/M));

opts.M = M;
opts.K = K;
opts.ind = ind;
opts.sigma = s;
opts.MaxIter = 100;

Pr0 = Pr;
err = zeros(1,opts.MaxIter);
for iter = 1:opts.MaxIter
  ticid=tic;
  Pr_ant = Pr;
  [Pr, Pc, w] = BayesSegIter(Y,Pr,opts);
  t = toc(ticid);
  
  err(iter) = norm(Pr-Pr_ant);
  plot(1:iter,err(1:iter))
  drawnow
  fprintf('Iteration %d/%d. Time %f\n',iter,opts.MaxIter,t)
end

[~,L1] = max(Pr0,[],2);
L1 = reshape(L1,siz);
L1(L1==4) = 0;
[~,L2] = max(Pr,[],2);
L2 = reshape(L2,siz);
L2(L2==4) = 0;
[~,L3] = max(Pc,[],2);
L3 = reshape(L3,siz);
L3(L3==4) = 0;
V.fname = 'Pr_1.nii';
spm_write_vol(V,L1);
V.fname = 'Pos_1.nii';
spm_write_vol(V,L2);
V.fname = 'Pc_1.nii';
spm_write_vol(V,L3);

A = kExpQuad(squareform(d),s,'distances');
P = w'*A;
stem(x,P')

% [V,D] = eigs(A,10);
% Y = A*V;
% plotmatrix(Y)