function E = BayesEnergy(Pr,Pc)

E = -sum(log(sum(Pr.*Pc,2)));