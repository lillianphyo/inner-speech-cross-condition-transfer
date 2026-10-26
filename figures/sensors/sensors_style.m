function S = sensors_style()
  set(0, 'defaultaxesfontsize', 8, 'defaulttextfontsize', 8, 'defaultaxesfontname', 'Helvetica', 'defaulttextfontname', 'Helvetica', 'defaultaxeslinewidth', 0.5, 'defaultaxesticklength', [0.012 0.012], 'defaultaxesbox', 'off');
  S.overt = [0 0.447 0.698];
  S.inner = [0.902 0.624 0];
  S.transfer = [0 0.620 0.451];
  S.shuffle = [0.6 0.6 0.6];
  S.purple = [0.8 0.475 0.655];
  S.sky = [0.337 0.706 0.914];
  S.vermilion = [0.835 0.369 0];
  S.dot = [0.25 0.25 0.25];
  S.chance = 0.25;
  S.here = fileparts(mfilename('fullpath'));
  S.runs = fullfile(S.here, '..', '..', 'runs');
end
