addpath(fileparts(mfilename('fullpath')));
S = sensors_style();
W = read_runs(fullfile(S.runs, 'honest_transfer'));

models = {'EEGNet', 'ShallowConvNet'};
windows = {'full', 'action', 'cue'};
arms = {'transfer', 'reciprocal', 'combined', 'mmd', 'shuffle_transfer', 'target_only'};
arm_labels = {'overt\rightarrowinner transfer', 'inner\rightarrowovert (reciprocal)', 'combined training', 'transfer + MMD', 'shuffled transfer', 'inner-only (100% calibration)'};
cols = {S.transfer, S.sky, S.overt, S.purple, S.shuffle, S.inner};
fractions = {'0.25', '0.5', '1.0'};
frac_x = [0.25 0.5 1.0];
win_cols = {S.overt, S.vermilion, S.transfer};
win_markers = {'o', 's', '^'};

fig_w = 17; fig_h = 11;
f = new_fig(fig_w, fig_h);

ax1 = axes('position', [1.2 / fig_w, 6.5 / fig_h, 15.55 / fig_w, 2.9 / fig_h]);
hold(ax1, 'on');
gap = 0.6;
set(ax1, 'xlim', [0.4, 6.6 + gap], 'ylim', [0 0.4], 'xtick', [], 'ytick', 0:0.1:0.4);
slope = hatch_slope(ax1);
bw = 0.13;
printf('%-15s %-7s %-17s mean [95%% t-CI]\n', 'model', 'window', 'arm');
for mi = 1:numel(models)
  for wi = 1:numel(windows)
    g = (mi - 1) * 3 + wi + (mi - 1) * gap;
    for ai = 1:numel(arms)
      m = subj_means(W, pick(W, 'window', windows{wi}, 'model', models{mi}, 'arm', arms{ai}, 'train_fraction', '1.0'));
      printf('%-15s %-7s %-17s %.4f [%.3f,%.3f]\n', models{mi}, windows{wi}, arms{ai}, mean(m), mean(m) - ci95(m), mean(m) + ci95(m));
      x = g + (ai - 3.5) * (bw + 0.01);
      patch(ax1, [x - bw / 2, x + bw / 2, x + bw / 2, x - bw / 2], [0 0 mean(m) mean(m)], cols{ai}, 'edgecolor', cols{ai}, 'linewidth', 0.5);
      errbar(ax1, x, mean(m), ci95(m), 0.03);
    end
    text(ax1, g, -0.012, [windows{wi} ' window'], 'horizontalalignment', 'center', 'verticalalignment', 'top', 'fontsize', 8);
  end
  gc = (mi - 1) * 3 + 2 + (mi - 1) * gap;
  text(ax1, gc, -0.075, models{mi}, 'horizontalalignment', 'center', 'verticalalignment', 'top', 'fontsize', 8, 'fontweight', 'bold');
  line(ax1, [gc - 1.45, gc + 1.45], [-0.058 -0.058], 'color', 'k', 'linewidth', 0.5, 'clipping', 'off');
end
chance_line(ax1, false);
ylabel(ax1, 'Accuracy');
panel_label(ax1, 'a', 'transfer arms and controls, 100% inner calibration data');

printf('\n%-15s %-7s fraction  mean [95%% t-CI]\n', 'model', 'window');
for mi = 1:numel(models)
  ax = axes('position', [(1.2 + (mi - 1) * 8.4) / fig_w, 1.45 / fig_h, 7.15 / fig_w, 3.4 / fig_h]);
  hold(ax, 'on');
  set(ax, 'xlim', [0.1 1.12], 'ylim', [0.1 0.4], 'xtick', frac_x, 'xticklabel', {'25%', '50%', '100%'}, 'ytick', 0.1:0.1:0.4);
  handles = [];
  for wi = 1:numel(windows)
    mu = zeros(1, 3); h = zeros(1, 3);
    for fi = 1:3
      m = subj_means(W, pick(W, 'window', windows{wi}, 'model', models{mi}, 'arm', 'target_only', 'train_fraction', fractions{fi}));
      mu(fi) = mean(m); h(fi) = ci95(m);
      printf('%-15s %-7s %-8s  %.4f [%.3f,%.3f]\n', models{mi}, windows{wi}, fractions{fi}, mu(fi), mu(fi) - h(fi), mu(fi) + h(fi));
    end
    xo = frac_x + (wi - 2) * 0.018;
    for fi = 1:3
      line(ax, [xo(fi) xo(fi)], [mu(fi) - h(fi), mu(fi) + h(fi)], 'color', win_cols{wi}, 'linewidth', 0.6);
    end
    handles(wi) = plot(ax, xo, mu, ['-' win_markers{wi}], 'color', win_cols{wi}, 'markerfacecolor', win_cols{wi}, 'markeredgecolor', win_cols{wi}, 'markersize', 4, 'linewidth', 0.8);
  end
  chance_line(ax, false);
  xlabel(ax, 'Inner-speech training data used');
  if mi == 1
    ylabel(ax, 'Inner-only accuracy');
  else
    set(ax, 'yticklabel', []);
  end
  if mi == 1
    win_labels = {'full window', 'action window', 'cue window'};
    for wi = 1:3
      yl = 0.385 - (wi - 1) * 0.025;
      plot(ax, [0.14 0.22], [yl yl], ['-' win_markers{wi}], 'color', win_cols{wi}, 'markerfacecolor', win_cols{wi}, 'markeredgecolor', win_cols{wi}, 'markersize', 4, 'linewidth', 0.8);
      text(ax, 0.25, yl, win_labels{wi}, 'verticalalignment', 'middle', 'fontsize', 7);
    end
  end
  panel_label(ax, char('a' + mi), [models{mi} ', inner\rightarrowinner calibration curve']);
end

ov = axes('position', [0 0 1 1], 'visible', 'off', 'xlim', [0 fig_w], 'ylim', [0 fig_h]);
hold(ov, 'on');
sw = 0.4; sh = 0.26;
y1 = fig_h - 0.5; y2 = fig_h - 0.92;
xs = [1.2 6.0 10.4];
for ai = 1:numel(arms)
  row = ceil(ai / 3); x = xs(ai - (row - 1) * 3);
  y = y1 - (row - 1) * (y1 - y2);
  patch(ov, [x x + sw x + sw x], [y y y + sh y + sh], cols{ai}, 'edgecolor', cols{ai});
  text(ov, x + sw + 0.12, y + sh / 2, arm_labels{ai}, 'verticalalignment', 'middle', 'fontsize', 8);
end
text(ov, fig_w - 0.25, 0.08, 'bars/markers: mean, whiskers: 95% t-CI over n = 10 subjects; dashed: chance = 0.25', 'horizontalalignment', 'right', 'verticalalignment', 'bottom', 'fontsize', 7, 'color', [0.3 0.3 0.3]);

save_fig(f, S.here, 'fig5_transfer_arms');
