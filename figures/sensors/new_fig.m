function f = new_fig(w_cm, h_cm)
  f = figure('visible', 'off', 'color', 'w');
  set(f, 'paperunits', 'centimeters', 'papersize', [w_cm h_cm]);
  set(f, 'paperpositionmode', 'manual', 'paperposition', [0.005 0.005 w_cm - 0.01 h_cm - 0.01]);
  set(f, 'units', 'centimeters', 'position', [0 0 w_cm h_cm]);
end
