function opts = BayesNormalMStep(Y,B,Q,opts)

C   = size(Q,2);
mu  = zeros(1,C);
cov = zeros(1,C);
gam = opts.mixture;

gB = B*gam';

parfor c = 1:C
  den    = sum(Q(:,c));
  mu(c)  = sum(Y.*Q(:,c))/den;  
  cov(c) = sum(Q(:,c).*(mu(c)-Y).^2)/den;
  gam(c) = den/sum(B(:,c)./gB);
end

opts.mixture    = gam/sum(gam);
opts.means      = mu;
opts.covariance = cov;