function errbar(ax, x, m, h, cap)
  line(ax, [x x], [m - h, m + h], 'color', 'k', 'linewidth', 0.6);
  line(ax, [x - cap, x + cap], [m - h, m - h], 'color', 'k', 'linewidth', 0.6);
  line(ax, [x - cap, x + cap], [m + h, m + h], 'color', 'k', 'linewidth', 0.6);
end
