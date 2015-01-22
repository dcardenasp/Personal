function G = transMatrix(numClasses, alpha, beta)

G = zeros(numClasses);
for c1=1:numClasses
for c2=1:numClasses
  if abs(c1-c2)==1
      G(c1,c2) = alpha;
  elseif abs(c1-c2)>1
      G(c1,c2) = beta;
  end
end
end