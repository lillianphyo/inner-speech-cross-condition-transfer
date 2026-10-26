function panel_label(ax, letter, name)
  text(ax, -0.06, 1.03, letter, 'units', 'normalized', 'fontweight', 'bold', 'fontsize', 9, 'verticalalignment', 'bottom', 'horizontalalignment', 'left');
  if nargin > 2
    text(ax, 0.08, 1.03, name, 'units', 'normalized', 'fontsize', 8, 'verticalalignment', 'bottom', 'horizontalalignment', 'left');
  end
end
