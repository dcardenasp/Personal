function [Q,E] = BayesNormalEStep(Y,B,opts)

N   = size(B,1);
C   = size(B,2);
Q   = zeros([N C]);
mu  = opts.means;
cov = opts.covariance;
gam = opts.mixture;
regularization = opts.mrf;

if regularization
  Pc = mrfRegularization(B,Q,opts);  
else
  parfor c = 1:C
    Pc(:,c) = gam(c)*B(:,c);
  end
end
z  = sum(Pc,2);
parfor c = 1:C    
  Pc(:,c) = Pc(:,c)./z;
end

parfor c = 1:C
  Q(:,c) = Pc(:,c).*mvnpdf(Y,mu(c),cov(c));
end
Q1  = sum(Q,2);
E   = -sum(log(Q1),1);
parfor c = 1:C
  Q(:,c) = Q(:,c)./Q1;
end