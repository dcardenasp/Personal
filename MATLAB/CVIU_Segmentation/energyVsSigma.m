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

blockwise = false;
V = spm_vol('T1_1.nii');
K  = 1e3;
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

Ind = randperm(numel(X),K);
x = X(Ind)';
X = X(:);
d = pdist(x);
s0 = kScaleOptimization(d);
Y = [X;ones(ceil(N/M)*M-N,1)];
Y = reshape(Y,M,ceil(N/M));

opts.M = M;
opts.K = K;
opts.indices = Ind;
opts.MaxIter = 1;
opts.minEnergyDiff = 1e-4;

S = linspace(s0/2,2*s0,100);

ener = zeros(1,numel(S));
for s = 1:numel(S)
  opts.sigma = S(s);
  Pr0  = Pr;
  ticid=tic;  
  [Pr, Pc, w] = BayesSegIter(Y,Pr0,opts);
  t = toc(ticid)
  ener(s) = BayesEnergy(Pr,Pc);
  plot(S(1:s),ener(1:s))
  drawnow
end
  