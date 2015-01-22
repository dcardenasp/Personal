function E = BayesEnergy(Pr,Pc)
if nargin == 1
  E = -sum(log(sum(Pr,2)));
elseif nargin == 2
  E = -sum(log(sum(Pr.*Pc,2)));
end