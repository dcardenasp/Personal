function [Pos, Pc, w] = patchBayesSegIter(X,Pr,Regs,opts)

N = numel(X);
R = unique(Regs(:));

s   = opts.sigma;
Ind = opts.indices;
K   = opts.K;
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
for n = 1:N
  r=find(Regs(n)==R);
  Pc(n,:) = kExpQuad2(X(n),x{r},s{r})*w{r};
end

Pos = Pc.*Pr;
Pos = Pos./repmat(sum(Pos,2),[1 size(Pc,2)]);  