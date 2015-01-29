gpuD = gpuDevice();
reset(gpuD)
gpuGauss = parallel.gpu.CUDAKernel('simpleEx.ptx','simpleEx.cu');
gpuGauss.ThreadBlockSize = 1024;
gpuGauss.GridSize = 3807;
F = gpuArray(zeros(sum(ind),1));
wait(gpuD)
tic
for i=1:10
  tmp = feval(gpuGauss,X(ind),X(Ind),w(1:100 + (i-1)*100,1),s,100);
  F = F + tmp;
end
toc


F = gpuArray(zeros(size(X)));
blk = 1024;
tic
for i=1:N/1024
  tmp = feval(gpuGauss,X(1:blk + (i-1)*blk),X(Ind),w(:,1),s,K);
  F(1:blk + (i-1)*blk) = tmp;
  %F(1:blk + (i-1)*blk) = gather(tmp);
end
toc