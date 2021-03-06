function [w2,cov2]= BayesParzenMStep(Y,F,Q,opts)

step = opts.step;
x = opts.samples;
K = opts.K;
cov = opts.covariance;
w = opts.mixture;
N = size(Y,1);
%C = opts.numClasses;
%blk = 1000;
QF=Q./F;

w_grad = zeros(size(w));
parfor k = 1:K
  w_grad(k,:) = mvnpdf(x(k,:),Y,cov)'*QF;
end
w2 = w + step*w_grad;
w2 = w2./repmat(sum(w2,1),[K 1]);

wkdd = zeros([1 1 N]);
parfor r = 1:N
%   f = mvnpdf(Y(r,:),x,cov);
  d = (x-Y(r,:));
%   kdd = f*(d'*d);
%   wkdd(r,:) = kdd'*w;
%   wkdd(:,:,r) = sum((d.*d)'*w);
  wkdd(:,:,r) = mean(d.*d);
end
% cov2 = 0.5*sum(sum(wkdd.*QF,1),2)/N;
cov2 = mean(wkdd,3);

% gpuD = gpuDevice(1);
% reset(gpuD)
% gpuGauss = parallel.gpu.CUDAKernel('simpleEx.ptx','simpleEx.cu');
% gpuGauss.ThreadBlockSize = 1024;
% gpuGauss.GridSize = ceil(K/1024);
% w_grad = zeros(size(w));
% for c=1:C
%   w_tmp = gpuArray(zeros(K,1));  
%   for i=1:ceil(N/blk)
%     ind = 1:blk + (i-1)*blk;
%     ind = ind(ind<=N);
%     tmp = feval(gpuGauss,x/sqrt(2*cov),Y(ind,:)/sqrt(2*cov),Q(ind,c),blk);
%     w_tmp = w_tmp + tmp;
%   end
%   w_grad(:,c) = gather(w_tmp);
% end
% w_grad = w_grad/sqrt(2*pi)/cov;