function F = BayesLikelihood(Y,opts)

N = size(Y,1);
x = opts.samples;
cov = opts.covariance;
C = opts.numClasses;
w = opts.mixture;
blk = 100;

gpuD = gpuDevice(1);
reset(gpuD)
gpuGauss = parallel.gpu.CUDAKernel('simpleEx.ptx','simpleEx.cu');
gpuGauss.ThreadBlockSize = 1024;
gpuGauss.GridSize = ceil(N/1024);
F = zeros(N,C);
for c=1:C
  F1 = gpuArray(zeros(N,1));
  %tmp = feval(gpuGauss,Y/sqrt(2*cov),x/sqrt(2*cov),w(:,c),opts.K);
  for i=1:opts.K/blk
    tmp = feval(gpuGauss,Y/sqrt(2*cov),x(1:blk + (i-1)*blk,:)/sqrt(2*cov),w(1:blk + (i-1)*blk,c),blk);
    F1 = F1 + tmp;
  end
  F(:,c) = gather(F1);
end
F = F/sqrt(2*pi)/cov;