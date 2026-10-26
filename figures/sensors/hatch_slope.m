function k = hatch_slope(ax)
  pos = get(ax, 'position');
  pp = get(ancestor(ax, 'figure'), 'paperposition');
  k = diff(get(ax, 'ylim')) * pos(3) * pp(3) / (diff(get(ax, 'xlim')) * pos(4) * pp(4));
end
