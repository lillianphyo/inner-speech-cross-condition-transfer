function draw_bar(ax, x0, x1, y, col, hatched, slope)
  if hatched
    patch(ax, [x0 x1 x1 x0], [0 0 y y], [1 1 1], 'edgecolor', col, 'linewidth', 0.6);
    step = (x1 - x0) / 3.5;
    for xs = (x0 - y / slope):step:x1
      xa = max(x0, xs);
      xb = min(x1, xs + y / slope);
      if xa < xb
        line(ax, [xa xb], slope * ([xa xb] - xs), 'color', col, 'linewidth', 0.6);
      end
    end
  else
    patch(ax, [x0 x1 x1 x0], [0 0 y y], col, 'edgecolor', col, 'linewidth', 0.6);
  end
end
