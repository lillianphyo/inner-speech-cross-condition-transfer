addpath(fileparts(mfilename('fullpath')));
S = sensors_style();
W = read_runs(fullfile(S.runs, 'honest_transfer'));
H = read_runs(fullfile(S.runs, 'session_split'));
L = read_runs(fullfile(S.runs, 'loso'));

models = {'EEGNet', 'ShallowConvNet', 'CSP+LDA', 'Riemannian'};
arms = struct( ...
  'label', {'overt\rightarrowovert', 'inner\rightarrowinner', 'overt\rightarrowinner', 'shuffled inner'}, ...
  'within', {'source_only', 'target_only', 'transfer', 'shuffle_target'}, ...
  'session', {'source_only', 'target_only', 'transfer', ''}, ...
  'loso', {'overt_loso', 'inner_loso', 'transfer_loso', 'shuffle_inner_loso'}, ...
  'col', {S.overt, S.inner, S.transfer, S.shuffle});
light = @(c) 0.45 * c + 0.55;

fig_w = 17; fig_h = 7.7;
f = new_fig(fig_w, fig_h);
left = 1.25; right = 0.25; gap = 0.3; bottom = 1.75; top = 1.7;
ax_w = (fig_w - left - right - 3 * gap) / 4;
ax_h = fig_h - bottom - top;
ymax = 0.62;
dx = 0.28; hw = 0.1;

printf('%-15s %-16s within(recomputed)  session(recomputed)  loso(recomputed)\n', 'model', 'arm');
for mi = 1:numel(models)
  ax = axes('position', [(left + (mi - 1) * (ax_w + gap)) / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h]);
  hold(ax, 'on');
  set(ax, 'xlim', [0.4 4.6], 'ylim', [0 ymax], 'xtick', [], 'ytick', 0:0.1:0.6);
  slope = hatch_slope(ax);
  for ai = 1:numel(arms)
    mw = subj_means(W, pick(W, 'window', 'action', 'model', models{mi}, 'arm', arms(ai).within, 'train_fraction', '1.0'));
    ml = subj_means(L, pick(L, 'model', models{mi}, 'arm', arms(ai).loso));
    xw = ai - dx; xh = ai; xl = ai + dx;
    draw_bar(ax, xw - hw, xw + hw, mean(mw), arms(ai).col, false, slope);
    draw_bar(ax, xl - hw, xl + hw, mean(ml), arms(ai).col, true, slope);
    subj_dots(ax, xw, mw, 0.06, S.dot);
    subj_dots(ax, xl, ml, 0.06, S.dot);
    errbar(ax, xw, mean(mw), ci95(mw), 0.04);
    errbar(ax, xl, mean(ml), ci95(ml), 0.04);
    if isempty(arms(ai).session)
      mh = [];
      text(ax, xh, 0.01, 'n/a', 'rotation', 90, 'horizontalalignment', 'left', 'verticalalignment', 'middle', 'fontsize', 7, 'color', [0.4 0.4 0.4]);
      ses_txt = 'n/a';
    else
      mh = subj_means(H, pick(H, 'window', 'action', 'model', models{mi}, 'arm', arms(ai).session));
      patch(ax, [xh - hw, xh + hw, xh + hw, xh - hw], [0 0 mean(mh) mean(mh)], light(arms(ai).col), 'edgecolor', arms(ai).col, 'linewidth', 0.6);
      subj_dots(ax, xh, mh, 0.06, S.dot);
      errbar(ax, xh, mean(mh), ci95(mh), 0.04);
      ses_txt = sprintf('%.4f [%.3f,%.3f]', mean(mh), mean(mh) - ci95(mh), mean(mh) + ci95(mh));
    end
    printf('%-15s %-16s %.4f [%.3f,%.3f]  %-21s  %.4f [%.3f,%.3f]\n', models{mi}, arms(ai).within, mean(mw), mean(mw) - ci95(mw), mean(mw) + ci95(mw), ses_txt, mean(ml), mean(ml) - ci95(ml), mean(ml) + ci95(ml));
    if ai == 1
      text(ax, xw - hw, max([mw; mh; ml]) + 0.025, sprintf('%.2f | %.2f | %.2f', mean(mw), mean(mh), mean(ml)), 'horizontalalignment', 'left', 'fontsize', 7, 'fontweight', 'bold');
    end
    text(ax, ai, -0.015, arms(ai).label, 'rotation', 30, 'horizontalalignment', 'right', 'verticalalignment', 'top', 'fontsize', 8);
  end
  chance_line(ax, false);
  if mi == 1
    ylabel(ax, 'Accuracy');
  else
    set(ax, 'yticklabel', []);
  end
  panel_label(ax, char('a' + mi - 1), models{mi});
end

ov = axes('position', [0 0 1 1], 'visible', 'off', 'xlim', [0 fig_w], 'ylim', [0 fig_h]);
hold(ov, 'on');
sw = 0.45; sh = 0.3;
y1 = fig_h - 0.45; y2 = fig_h - 0.9;
x = 1.3;
patch(ov, [x x + sw x + sw x], [y1 y1 y1 + sh y1 + sh], [0.3 0.3 0.3], 'edgecolor', [0.3 0.3 0.3]);
text(ov, x + sw + 0.15, y1 + sh / 2, 'within-subject, random trial split (solid)', 'verticalalignment', 'middle', 'fontsize', 8);
x = 8.6;
patch(ov, [x x + sw x + sw x], [y1 y1 y1 + sh y1 + sh], light([0.3 0.3 0.3]), 'edgecolor', [0.3 0.3 0.3], 'linewidth', 0.6);
text(ov, x + sw + 0.15, y1 + sh / 2, 'within-subject, held-out session (light)', 'verticalalignment', 'middle', 'fontsize', 8);
x = 1.3;
patch(ov, [x x + sw x + sw x], [y2 y2 y2 + sh y2 + sh], [1 1 1], 'edgecolor', [0.3 0.3 0.3], 'linewidth', 0.6);
for xs = (x - sh):(sw / 3.5):(x + sw)
  xa = max(x, xs); xb = min(x + sw, xs + sh);
  if xa < xb, line(ov, [xa xb], y2 + ([xa xb] - xs), 'color', [0.3 0.3 0.3], 'linewidth', 0.6); end
end
text(ov, x + sw + 0.15, y2 + sh / 2, 'across-subject, LOSO (hatched)', 'verticalalignment', 'middle', 'fontsize', 8);
x = 8.6;
plot(ov, x + 0.2, y2 + sh / 2, 'o', 'markersize', 2.2, 'markerfacecolor', S.dot, 'markeredgecolor', S.dot);
text(ov, x + sw + 0.15, y2 + sh / 2, 'per-subject mean', 'verticalalignment', 'middle', 'fontsize', 8);
x = 12.6;
line(ov, [x x + sw], [y2 + sh / 2, y2 + sh / 2], 'color', 'k', 'linestyle', '--', 'linewidth', 0.6);
text(ov, x + sw + 0.15, y2 + sh / 2, 'chance = 0.25', 'verticalalignment', 'middle', 'fontsize', 8);
text(ov, fig_w - right, 0.12, 'bars: mean, whiskers: 95% t-CI over n = 10 subjects; action window; n/a: shuffled inner-only arm not run with held-out session', 'horizontalalignment', 'right', 'verticalalignment', 'bottom', 'fontsize', 7, 'color', [0.3 0.3 0.3]);

save_fig(f, S.here, 'fig2_within_vs_loso');
