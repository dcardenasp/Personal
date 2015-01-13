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

blockwise = true;
V = spm_vol('T1_1.nii');
K  = 3e3;
M = 5e3;
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
N = prod(siz);

if blockwise
  ind  = randperm(prod(floor(siz/B)),K);
  [j1,j2,j3]=ind2sub(floor(siz/B),ind);
  blkSiz = floor(siz/B);
  i=1;
  Ind = cell(B^3,1);
  s   = cell(B^3,1);
  Regs = zeros(siz);
  
  for b1=0:B-1
  for b2=0:B-1
  for b3=0:B-1
      
    i1 = b1*(blkSiz(1)+1)+(1:blkSiz(1)+1);
    i2 = b2*(blkSiz(2)+1)+(1:blkSiz(2)+1);
    i3 = b3*(blkSiz(3)+1)+(1:blkSiz(3)+1);
    [i1,i2,i3]=ndgrid(i1,i2,i3);
    i4 = sub2ind(siz, min(i1(:),siz(1)), min(i2(:),siz(2)), min(i3(:),siz(3)));
    Regs(i4) = i;
    ind = sub2ind(siz, b1*blkSiz(1)+j1, b2*blkSiz(2)+j2, b3*blkSiz(3)+j3);
    Ind{i} = ind;
    x = reshape(X(Ind{i}),[K 1]);
    d = pdist(x);
    s{i} = kScaleOptimization(d);
    i = i+1;
  end
  end
  end
else
  Ind = randperm(numel(X),K);
  x = X(Ind)';
  X = X(:);
  d = pdist(x);
  s = kScaleOptimization(d);
  Y = [X;ones(ceil(N/M)*M-N,1)];
  Y = reshape(Y,M,ceil(N/M));
end
opts.M = M;
opts.K = K;
opts.indices = Ind;
opts.sigma = s;
opts.MaxIter = 100;
opts.minEnergyDiff = 1e-4;

Pr0  = Pr;
err  = zeros(1,opts.MaxIter);
ener = zeros(1,opts.MaxIter);
for iter = 1:opts.MaxIter
  ticid=tic;
  Pr_ant = Pr;
  if blockwise
    [Pr, Pc, w] = patchBayesSegIter(X,Pr,Regs,opts);
  else
    [Pr, Pc, w] = BayesSegIter(Y,Pr,opts);
  end
  t = toc(ticid);
  
  err(iter) = norm(Pr-Pr_ant);
  
  ener(iter) = BayesEnergy(Pr,Pc);
    
  plotyy(1:iter,err(1:iter),1:iter,ener(1:iter))
  drawnow
  
  ed = inf;
  if iter>2    
    ed = abs(ener(iter)-ener(iter-1));
    if ed<opts.minEnergyDiff
        break;
    end
  end
  
  fprintf('Iteration %d/%d. ED %f. Time %f\n',iter,opts.MaxIter,ed,t)  
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
V.fname = 'Pr_2.nii';
spm_write_vol(V,L1);
V.fname = 'Pos_2.nii';
spm_write_vol(V,L2);
V.fname = 'Pc_2.nii';
spm_write_vol(V,L3);

A = kExpQuad(squareform(d),s,'distances');
P = w'*A;
stem(x,P')

% [V,D] = eigs(A,10);
% Y = A*V;
% plotmatrix(Y)