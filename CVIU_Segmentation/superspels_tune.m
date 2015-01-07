clear all
close all
clc

addpath(genpath('/home/dcardenasp/Documents/MATLAB/spm8'))
addpath(genpath('/home/dcardenasp/Documents/MATLAB/myImTools'))

T = spm_read_vols(spm_vol('/home/dcardenasp/supervoxels_slic/T1.nii'));

N = [1e2:1e2:1e3 1e3:1e3:1e4];
C = 0.1:0.1:1;

nmi = zeros(numel(N),numel(C));

for n = 1:numel(N)
for c = 1:numel(C)
  fprintf('%d-%d, %d-%d\n',n,numel(N),c,numel(C))
  name = ['/home/dcardenasp/supervoxels_slic/T1_' num2str(C(c)) '_' num2str(N(n)) '.nii'];
  fid = fopen(['/home/dcardenasp/supervoxels_slic/T1_' num2str(C(c)) '_' num2str(N(n)) '.nii']);
  if fid~=-1
    fclose(fid);
    S = spm_read_vols(spm_vol(name));
    [~,~,nmi(n,c)] = mutualinformation(T,S);    
  end
end
end