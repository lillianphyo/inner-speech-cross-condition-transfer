addpath(fileparts(mfilename('fullpath')));
S = sensors_style();
C = read_runs(fullfile(S.runs, 'overt_channels'));
B = read_runs(fullfile(S.runs, 'honest_transfer'));
N = read_runs(fullfile(S.runs, 'honest_transfer_8_30'));
G = read_runs(fullfile(S.runs, 'honest_transfer_30_100'));

fig_w = 19.5; fig_h = 8.4;
f = new_fig(fig_w, fig_h);
bottom = 2.0; top = 1.6; ax_h = fig_h - bottom - top;
ymax = 0.75;

conditions = {'all', 'peripheral', 'left_sensorimotor', 'right_sensorimotor', 'random_9', 'left_primary_strip', 'right_primary_strip', 'random_3'};
cond_labels = {'all (128)', 'peripheral (16)', 'left sensorimotor (9)', 'right sensorimotor (9)', 'random 9 (3 draws)', 'left primary strip (3)', 'right primary strip (3)', 'random 3 (3 draws)'};

function [m, draws] = cond_means(T, cond, arm)
  if any(strcmp(cond, {'random_9', 'random_3'}))
    mask = pick(T, 'arm', arm) & strncmp(T.condition, [cond '_'], numel(cond) + 1);
  else
    mask = pick(T, 'condition', cond, 'arm', arm);
  end
  draws = numel(unique(T.condition(mask)));
  m = subj_means(T, mask);
end

ax1 = axes('position', [1.2 / fig_w, bottom / fig_h, 7.4 / fig_w, ax_h / fig_h]);
hold(ax1, 'on');
nc = numel(conditions);
set(ax1, 'xlim', [0.4 nc + 0.6], 'ylim', [0 ymax], 'xtick', [], 'ytick', 0:0.1:0.7);
slope = hatch_slope(ax1);
printf('%-22s overt(recomputed)      inner(recomputed)\n', 'condition');
for ci = 1:nc
  [mo, d] = cond_means(C, conditions{ci}, 'source_only');
  mi_ = cond_means(C, conditions{ci}, 'target_only');
  printf('%-22s %.4f [%.3f,%.3f]  %.4f [%.3f,%.3f]  draws=%d\n', conditions{ci}, mean(mo), mean(mo) - ci95(mo), mean(mo) + ci95(mo), mean(mi_), mean(mi_) - ci95(mi_), mean(mi_) + ci95(mi_), d);
  draw_bar(ax1, ci - 0.38, ci - 0.02, mean(mo), S.overt, false, slope);
  draw_bar(ax1, ci + 0.02, ci + 0.38, mean(mi_), S.inner, false, slope);
  subj_dots(ax1, ci - 0.2, mo, 0.1, S.dot);
  subj_dots(ax1, ci + 0.2, mi_, 0.1, S.dot);
  errbar(ax1, ci - 0.2, mean(mo), ci95(mo), 0.06);
  errbar(ax1, ci + 0.2, mean(mi_), ci95(mi_), 0.06);
  text(ax1, ci, -0.015, cond_labels{ci}, 'rotation', 30, 'horizontalalignment', 'right', 'verticalalignment', 'top', 'fontsize', 8);
end
chance_line(ax1, false);
ylabel(ax1, 'Accuracy');
panel_label(ax1, 'a', 'EEGNet, channel subsets');

models = {'EEGNet', 'ShallowConvNet', 'CSP+LDA', 'Riemannian'};
arms = {'source_only', 'target_only', 'transfer'};
shuf = {'shuffle_source', 'shuffle_target', 'shuffle_transfer'};
cols = {S.overt, S.inner, S.transfer};
bands = struct('name', {'broadband', '8-30 Hz', '30-100 Hz'}, 'T', {B, N, G}, 'dx', {-0.095, 0, 0.095});

function m = band_means(T, model, arm)
  mask = pick(T, 'window', 'action', 'model', model, 'arm', arm, 'train_fraction', '1.0');
  m = subj_means(T, mask);
end

ax2 = axes('position', [9.6 / fig_w, bottom / fig_h, 9.6 / fig_w, ax_h / fig_h]);
hold(ax2, 'on');
set(ax2, 'xlim', [0.4 4.6], 'ylim', [0 ymax], 'xtick', [], 'ytick', 0:0.1:0.7, 'yticklabel', []);
slope = hatch_slope(ax2);
bw = 0.08;
shuf_col = [0.45 0.45 0.45];
printf('%-15s %-12s broadband             8-30 Hz               30-100 Hz             shuffle broad / 8-30 / 30-100\n', 'model', 'arm');
for mi = 1:numel(models)
  for ai = 1:numel(arms)
    xc = mi + (ai - 2) * 0.32;
    means = zeros(1, 3); tops = zeros(1, 3); cells = cell(1, 3); shuf_txt = cell(1, 3);
    for bi = 1:3
      T = bands(bi).T;
      m = band_means(T, models{mi}, arms{ai});
      means(bi) = mean(m); tops(bi) = mean(m) + ci95(m);
      cells{bi} = sprintf('%.4f [%.3f,%.3f]', mean(m), mean(m) - ci95(m), mean(m) + ci95(m));
      x = xc + bands(bi).dx;
      switch bi
        case 1, draw_bar(ax2, x - bw / 2, x + bw / 2, mean(m), cols{ai}, false, slope);
        case 2, patch(ax2, [x - bw / 2, x + bw / 2, x + bw / 2, x - bw / 2], [0 0 mean(m) mean(m)], [1 1 1], 'edgecolor', cols{ai}, 'linewidth', 0.9);
        case 3, draw_bar(ax2, x - bw / 2, x + bw / 2, mean(m), cols{ai}, true, slope);
      end
      errbar(ax2, x, mean(m), ci95(m), 0.02);
      if any(pick(T, 'model', models{mi}, 'arm', shuf{ai}))
        sm = band_means(T, models{mi}, shuf{ai});
        line(ax2, [x - bw / 2 - 0.015, x + bw / 2 + 0.015], [mean(sm) mean(sm)], 'color', shuf_col, 'linewidth', 1.4);
        shuf_txt{bi} = sprintf('%.3f', mean(sm));
      else
        shuf_txt{bi} = 'n/a';
      end
    end
    printf('%-15s %-12s %s  %s  %s  %s / %s / %s\n', models{mi}, arms{ai}, cells{:}, shuf_txt{:});
    if ai == 1
      text(ax2, xc + bands(1).dx - bw / 2, max(tops) + 0.015, sprintf('%.2f\\rightarrow%.2f\\rightarrow%.2f', means), 'horizontalalignment', 'left', 'fontsize', 7, 'fontweight', 'bold');
    end
  end
  text(ax2, mi, -0.015, models{mi}, 'rotation', 30, 'horizontalalignment', 'right', 'verticalalignment', 'top', 'fontsize', 8);
end
chance_line(ax2, false);
panel_label(ax2, 'b', 'all 128 channels, band-pass');

ov = axes('position', [0 0 1 1], 'visible', 'off', 'xlim', [0 fig_w], 'ylim', [0 fig_h]);
hold(ov, 'on');
sw = 0.4; sh = 0.28;
y1 = fig_h - 0.5; y2 = fig_h - 0.95;
items = {1.2, S.overt, 'overt-only (train overt, test overt)'; 6.6, S.inner, 'inner-only'; 8.8, S.transfer, 'overt\rightarrowinner transfer'};
for k = 1:3
  x = items{k, 1};
  patch(ov, [x x + sw x + sw x], [y1 y1 y1 + sh y1 + sh], items{k, 2}, 'edgecolor', items{k, 2});
  text(ov, x + sw + 0.12, y1 + sh / 2, items{k, 3}, 'verticalalignment', 'middle', 'fontsize', 8);
end
x = 1.2;
patch(ov, [x x + sw x + sw x], [y2 y2 y2 + sh y2 + sh], [0.3 0.3 0.3], 'edgecolor', [0.3 0.3 0.3]);
text(ov, x + sw + 0.12, y2 + sh / 2, 'filled: broadband', 'verticalalignment', 'middle', 'fontsize', 8);
x = 4.4;
patch(ov, [x x + sw x + sw x], [y2 y2 y2 + sh y2 + sh], [1 1 1], 'edgecolor', [0.3 0.3 0.3], 'linewidth', 0.9);
text(ov, x + sw + 0.12, y2 + sh / 2, 'hollow: 8-30 Hz', 'verticalalignment', 'middle', 'fontsize', 8);
x = 7.4;
patch(ov, [x x + sw x + sw x], [y2 y2 y2 + sh y2 + sh], [1 1 1], 'edgecolor', [0.3 0.3 0.3], 'linewidth', 0.6);
for xs = (x - sh):(sw / 3.5):(x + sw)
  xa = max(x, xs); xb = min(x + sw, xs + sh);
  if xa < xb, line(ov, [xa xb], y2 + ([xa xb] - xs), 'color', [0.3 0.3 0.3], 'linewidth', 0.6); end
end
text(ov, x + sw + 0.12, y2 + sh / 2, 'hatched: 30-100 Hz', 'verticalalignment', 'middle', 'fontsize', 8);
x = 11.0;
line(ov, [x x + sw], [y2 + sh / 2, y2 + sh / 2], 'color', shuf_col, 'linewidth', 1.4);
text(ov, x + sw + 0.12, y2 + sh / 2, 'shuffled-label control of the same arm', 'verticalalignment', 'middle', 'fontsize', 8);
text(ov, fig_w - 0.3, 0.12, 'action window; bars: mean, whiskers: 95% t-CI over n = 10 subjects; dashed: chance = 0.25', 'horizontalalignment', 'right', 'verticalalignment', 'bottom', 'fontsize', 7, 'color', [0.3 0.3 0.3]);

save_fig(f, S.here, 'fig4_channels_band');
