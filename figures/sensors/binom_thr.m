function thr = binom_thr(n, p)
  k = 0:n;
  logpmf = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * log(p) + (n - k) * log(1 - p);
  cdf = cumsum(exp(logpmf));
  thr = k(find(cdf >= 0.95, 1)) / n;
end
