function chance_line(ax, label)
  xl = get(ax, 'xlim');
  line(ax, xl, [0.25 0.25], 'color', 'k', 'linestyle', '--', 'linewidth', 0.6);
  if nargin > 1 && label
    text(ax, xl(2), 0.25, ' chance', 'verticalalignment', 'middle', 'horizontalalignment', 'left', 'fontsize', 7);
  end
end
