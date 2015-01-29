function [w,cov]= BayesParzenMStep(Y,F,Q,opts)

step = opts.step;
x = opts.samples;
K = opts.K;
cov = opts.covariance;
w = opts.mixture;
N = size(Y,1);
%C = opts.numClasses;
%blk = 1000;
Q=Q./F;

w_grad = zeros(size(w));
parfor k = 1:K
  w_grad(k,:) = mvnpdf(x(k,:),Y,cov)'*Q;
end
w = w - step*w_grad;
w = w./repmat(sum(w,1),[K 1]);

wkdd = zeros(size(Q));
wf = zeros(size(Q));
parfor r = 1:N
  f = mvnpdf(Y(r,:),x,cov);
  d = (x-Y(r,:));
  kdd = f*(d'*d);
  wkdd(r,:) = kdd'*w;
  wf(r,:) = f'*w;
end
cov = 0.5*sum(sum(wkdd.*Q,1),2)./sum(sum(wf.*Q,1),2);

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