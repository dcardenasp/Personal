clear all
close all
clc

addpath(genpath('/home/dcardenasp/Documents/MATLAB/MLmat/kernels'))
addpath(genpath('/home/dcardenasp/Documents/MATLAB/spm8'))
addpath(genpath('/home/dcardenasp/Documents/MATLAB/SliceBrowser'))

N = [1e2:1e2:1e3 2e3:1e3:1e4 2e4:1e4:10e4];
R = 30;

X=spm_read_vols(spm_vol('/home/dcardenasp/Documents/DataBases/ADNI/AD/002_S_0816/MPR____N3__Scaled/2006-09-29_14_09_26.0/S19532/ADNI_002_S_0816_MR_MPR____N3__Scaled_Br_20070217005105475_S19532_I40726.nii' ));

s = zeros(numel(N),R);
v = zeros(numel(N),R);

for r=1:R
    P = randperm(numel(X));
    for n=1:numel(N)
        fprintf('%d-%d %d-%d, %d\n',r,R,n,numel(N),N(n))
        x = X(P(1:N(n)))';
        d = pdist(x);
        [s(n,r),v(n,r)] = kScaleOptimization(d);        
    end
    figure(1)
    clf 
    subplot(2,1,1)
    errorbar(N,mean(s(:,1:r),2),std(s(:,1:r),[],2))
    subplot(2,1,2)
    errorbar(N,mean(v(:,1:r),2),std(v(:,1:r),[],2))
    drawnow
    pause(0.5)
    save data/sigma_stability.mat s v N
end

%%
N=10000;
ind = round(linspace(1,numel(X),N));
x = X(ind)';
[w,h,l] = ind2sub(size(X),ind);
d = pdist(x);
[sopt,vopt] = kScaleOptimization(d);

F = zeros(size(X));
for i=1:numel(X)
  fprintf('%d-%d\n',i,numel(X))
  for mu=x'
    F(i) = F(i)+normpdf(X(i),mu,sopt);
  end
end