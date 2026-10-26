function h = ci95(x)
  n = numel(x);
  df = n - 1;
  b = betaincinv(0.05, df / 2, 0.5);
  tcrit = sqrt(df * (1 - b) / b);
  h = tcrit * std(x) / sqrt(n);
end
