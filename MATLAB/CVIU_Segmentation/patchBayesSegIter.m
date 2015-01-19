function [Pos, Pc, w] = patchBayesSegIter(X,Pr,Regs,opts)

N = numel(X);
R = unique(Regs(:));

s   = opts.sigma;
Ind = opts.indices;
K   = opts.K;
M   = opts.M;
x = cell(numel(R),1);
for r = 1:numel(R)
  x{r} = reshape(X(Ind{r}),K,1);
end

w = cell(numel(R),1);
for r = 1:numel(R);
  w{r} = Pr(Ind{r},:);
  w{r} = w{r}./repmat(sum(w{r}),[K 1]);
  w{r}(isnan(w{r})) = 0;
end

Pc = zeros(size(Pr));
for r = 1:numel(R)
  Nr = sum(Regs(:)==R(r));
  Y = reshape([X(Regs(:)==R(r));ones(ceil(Nr/M)*M-Nr,1)], M, ceil(Nr/M));
  tmp = zeros([size(Y) size(Pr,2)]);
  parfor i = 1:ceil(Nr/M)
    tmp(:,i,:) = kExpQuad2(Y(:,i),x{r},s{r})*w{r};
  end
  tmp=reshape(tmp,size(tmp,1)*size(tmp,2),size(tmp,3));
  Pc(Regs(:)==R(r),:) = tmp(1:Nr,:);  
end

Pos = Pc.*Pr;
Pos = Pos./repmat(sum(Pos,2)+eps,[1 size(Pc,2)]);  