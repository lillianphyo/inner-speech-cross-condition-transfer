function subj_dots(ax, x, vals, halfwidth, col)
  n = numel(vals);
  jit = linspace(-halfwidth, halfwidth, n);
  [~, order] = sort(vals);
  xs = x + jit(order);
  vals = vals(:)';
  plot(ax, xs, vals, 'o', 'markersize', 2.2, 'markeredgecolor', col, 'markerfacecolor', col, 'linewidth', 0.3);
end
