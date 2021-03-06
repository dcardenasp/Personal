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

method = 3; %0: parzen full. 1: parzen block. 2: parzen local. 3: Parzen coord as feature
regularization = true;
V = spm_vol('/media/dcardenasp/VERBATIM/DataBases/subject_01/T1_1.nii');
K  = 1000;
M = 5e3;
B = 3;
radius = 20;

y = (0:600)';

X = spm_read_vols(V);
siz = size(X);
N = prod(siz);
Pr = zeros([siz 6]);
list = [6 5 4 3 1 2];
parfor l=1:numel(list)
  Pr(:,:,:,l) = spm_read_vols(spm_vol(...
    ['/media/dcardenasp/VERBATIM/DataBases/subject_01/c' num2str(list(l)) '_1.nii']));
end
% Pr(:,:,:,2) = spm_read_vols(spm_vol('/media/dcardenasp/VERBATIM/DataBases/subject_01/c5_1.nii'));
% Pr(:,:,:,3) = spm_read_vols(spm_vol('/media/dcardenasp/VERBATIM/DataBases/subject_01/c4_1.nii'));
% Pr(:,:,:,4) = spm_read_vols(spm_vol('/media/dcardenasp/VERBATIM/DataBases/subject_01/c3_1.nii'));
% Pr(:,:,:,5) = spm_read_vols(spm_vol('/media/dcardenasp/VERBATIM/DataBases/subject_01/c1_1.nii'));
% Pr(:,:,:,6) = spm_read_vols(spm_vol('/media/dcardenasp/VERBATIM/DataBases/subject_01/c2_1.nii'));

numClasses  = size(Pr,4);
G = zeros(numClasses);
for c1=1:numClasses
for c2=1:numClasses
  if abs(c1-c2)==1
      G(c1,c2) = 0.5;
  elseif abs(c1-c2)>1
      G(c1,c2) = 3;
  end
end
end

Pr  = Pr - min(Pr(:));
sPr = sum(Pr,4);
Pr  = Pr./repmat(sPr,[1 1 1 size(Pr,4)]);
Pr  = reshape(Pr,prod(siz),size(Pr,4));

Ind = randperm(numel(X),K);
x = X(Ind)';
X = X(:);
d = pdist(x);
s0 = 80;%kScaleOptimization(d);

switch method
case 1
  ind = randperm(prod(floor(siz/B)),K);
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
%     s{i} = kScaleOptimization(d);
    s{i} = s0;
    i = i+1;
  end
  end
  end  
case 0
  s = s0;
  Y = [X;ones(ceil(N/M)*M-N,1)];
  Y = reshape(Y,M,ceil(N/M));
case 2 
  s = 1000;
case 3
  s = 1000;      
end

opts.M = M;
opts.K = K;
opts.indices = Ind;
opts.sigma = s;
opts.MaxIter = 100;
opts.minEnergyDiff = 1e-4;
opts.size = siz;
opts.radius = radius;
opts.Transition = G;
opts.covariance = [s^2 0 0 0; 0 5^2 0 0; 0 0 5^2 0; 0 0 0 5^2];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch method
case 0
    pr = Pr(Ind,:);
    w = pr./repmat(sum(pr),[K 1]);
    num = zeros(1,N);
    den = num;
    parfor r=1:N
        num(r) = ((X(r)-x').^2)*w*Pr(r,:)';
        den(r) = sum(w*Pr(r,:)');
    end
    opts.sigma = sqrt(sum(num)/sum(den));
case 1
    R = unique(Regs(:));
    for reg = 1:numel(R)
        pr = Pr(Ind{reg},:);
        w = pr./repmat(sum(pr),[K 1]);
        X2 = X(Regs(:)==R(reg));
        n = numel(X2);
        X2 = reshape(X2,[n 1]);
        num = zeros(1,n);
        parfor r=1:n
            num(r) = ((X(r)-x').^2)*w*Pr(r,:)';
        end
        s{reg} = sqrt(mean(num));
    end
    opts.sigma = s;
case 2
    w = Inf;
case 3
    w = Inf;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Pr0  = Pr;
err  = zeros(1,opts.MaxIter);
ener = zeros(1,opts.MaxIter);
if method == 1
    S = zeros(B^3,opts.MaxIter);
else
    S = zeros(1,opts.MaxIter);
end

for iter = 1:opts.MaxIter
  ticid=tic;
  Pr_ant = Pr;
  w_ant = w;
  if method == 1
    [Pr, Pc, w] = patchBayesSegIter(X,Pr,Regs,opts);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R = unique(Regs(:));
    for reg = 1:numel(R)
        pr = Pr(Ind{reg},:);
        w1 = pr./repmat(sum(pr),[K 1]);
        X2 = X(Regs(:)==R(reg));
        n = numel(X2);
        X2 = reshape(X2,[n 1]);
        num = zeros(1,n);
        parfor r=1:n
            num(r) = ((X(r)-x').^2)*w1*Pr(r,:)';    
        end
        s{reg} = sqrt(mean(num));
    end
    opts.sigma = s;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif method == 0 
    [Pr, Pc, w] = BayesSegIter(Y,Pr,opts);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    num = zeros(1,N);    
    parfor r=1:N
      num(r) = ((X(r)-x').^2)*w*Pr(r,:)';    
    end
    opts.sigma = sqrt(mean(num));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  elseif method == 2
    [Pr, Pc, w, s] = LocalBayesSegIter(X,Pr,opts);
    opts.sigma = s;
  elseif method == 3
    tic
    [Pr, Pc, w] = BayesSegSpaceIter(X,Pr,opts);
    toc
    opts.sigma = s;
  end
  if regularization
      Pr = mrfRegularization(Pr,opts);
  end
  t = toc(ticid);
  
  err(iter) = norm(Pr-Pr_ant);  
  ener(iter) = BayesEnergy(Pr,Pc);
  if method == 1
      S(:,iter) = cell2mat(opts.sigma);
  else
      S(iter)   = opts.sigma;
  end
          
  figure(1)
  plotyy(1:iter,S(:,1:iter),1:iter,ener(1:iter))
  drawnow
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  figure(2)
  if method == 1
    for i=1:B^3
      ind2=Ind{i};
      x = reshape(X(ind2),[K 1]);
      A = kExpQuad2(y,x,s{i});
      P = A*w{i};
      subplot(ceil(sqrt(B^3)),ceil(sqrt(B^3)),i)
      plot(y,P)
    end
  elseif method == 0
    A = kExpQuad2(y,X(Ind),s);
    pr = Pr(Ind,:);
    P = A*w;
    plot(y,P)    
  end
  drawnow
  
  figure(3)
  plot(S(1:iter),ener(1:iter))
  drawnow
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  ed = inf;
  wd = inf;
  if iter>2    
    ed = abs(ener(iter)-ener(iter-1));
    if ed<opts.minEnergyDiff
      break;
    end
    if ~iscell(w)
      wd = norm(w_ant-w);      
    else
      wd = norm(cell2mat(w_ant)-cell2mat(w));
    end
    if wd<opts.minEnergyDiff
      break;
    end
  end
  
  fprintf('Iteration %d/%d. ED %.2f. WD %.2e. Time %.2f\n',iter,opts.MaxIter,ed,wd,t)  
end

[~,L1] = max(Pr0,[],2);
L1 = reshape(L1,siz)-1;
[~,L2] = max(Pr,[],2);
L2 = reshape(L2,siz)-1;
[~,L3] = max(Pc,[],2);
L3 = reshape(L3,siz)-1;
V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/Pr_1.nii';
spm_write_vol(V,L1);
V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/Pos_1.nii';
spm_write_vol(V,L2);
V.fname = '/media/dcardenasp/VERBATIM/DataBases/subject_01/Pc_1.nii';
spm_write_vol(V,L3);

A = kExpQuad(squareform(d),s,'distances');
P = w'*A;
stem(x,P')

% [V,D] = eigs(A,10);
% Y = A*V;
% plotmatrix(Y)