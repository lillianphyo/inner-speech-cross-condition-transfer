function save_fig(f, here, stem)
  print(f, fullfile(here, [stem '.png']), '-dpng', '-r300');
  print(f, fullfile(here, [stem '.pdf']), '-dpdf');
  printf('wrote %s.{png,pdf}\n', fullfile(here, stem));
end
