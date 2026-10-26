addpath(fileparts(mfilename('fullpath')));
S = sensors_style();
W = read_runs(fullfile(S.runs, 'honest_transfer'));
L = read_runs(fullfile(S.runs, 'loso'));

models = {'EEGNet', 'ShallowConvNet', 'CSP+LDA', 'Riemannian'};
markers = {'o', 's', '^', 'd'};
cols = {S.overt, S.inner, S.transfer, S.purple};

fig_w = 8.5; fig_h = 8.5;
f = new_fig(fig_w, fig_h);
ax = axes('position', [1.3 / fig_w, 1.2 / fig_h, 6.9 / fig_w, 6.9 / fig_h]);
hold(ax, 'on');
lim = [0.1 0.65];
set(ax, 'xlim', lim, 'ylim', lim, 'xtick', 0.1:0.1:0.6, 'ytick', 0.1:0.1:0.6);

xw_all = []; yl_all = []; thr_w_all = []; thr_l_all = [];
pts = {}; labels = {};
printf('%-15s within  loso   above_within_thr  above_loso_thr  below_diag\n', 'model');
for mi = 1:numel(models)
  [xw, subj_w, nt_w] = subj_means(W, pick(W, 'window', 'action', 'model', models{mi}, 'arm', 'source_only', 'train_fraction', '1.0'));
  [yl, subj_l, nt_l] = subj_means(L, pick(L, 'model', models{mi}, 'arm', 'overt_loso'));
  if ~isequal(subj_w, subj_l), error('subject sets differ'); end
  thr_w = arrayfun(@(n) binom_thr(n, 0.25), nt_w);
  thr_l = arrayfun(@(n) binom_thr(n, 0.25), nt_l);
  xw_all = [xw_all; xw]; yl_all = [yl_all; yl]; thr_w_all = [thr_w_all; thr_w]; thr_l_all = [thr_l_all; thr_l];
  n_w = sum(xw > thr_w); n_l = sum(yl > thr_l);
  printf('%-15s %.4f %.4f  %d/10  %d/10  %d/10\n', models{mi}, mean(xw), mean(yl), n_w, n_l, sum(yl < xw));
  pts{mi} = [xw yl];
  labels{mi} = sprintf('%s  (%d | %d)', models{mi}, n_w, n_l);
end

bw = [min(thr_w_all) max(thr_w_all)]; bl = [min(thr_l_all) max(thr_l_all)];
patch(ax, [bw(1) bw(2) bw(2) bw(1)], [lim(1) lim(1) lim(2) lim(2)], [0.88 0.88 0.88], 'edgecolor', 'none');
patch(ax, [lim(1) lim(2) lim(2) lim(1)], [bl(1) bl(1) bl(2) bl(2)], [0.88 0.88 0.88], 'edgecolor', 'none');
for mi = 1:numel(models)
  handles(mi) = plot(ax, pts{mi}(:, 1), pts{mi}(:, 2), markers{mi}, 'markersize', 4.5, 'markeredgecolor', cols{mi}, 'markerfacecolor', cols{mi}, 'linewidth', 0.5, 'linestyle', 'none');
end
line(ax, lim, lim, 'color', [0.4 0.4 0.4], 'linewidth', 0.6);
line(ax, lim, [0.25 0.25], 'color', 'k', 'linestyle', '--', 'linewidth', 0.6);
line(ax, [0.25 0.25], lim, 'color', 'k', 'linestyle', '--', 'linewidth', 0.6);
text(ax, lim(2) - 0.01, lim(2) - 0.035, 'y = x', 'horizontalalignment', 'right', 'fontsize', 7, 'color', [0.4 0.4 0.4]);
text(ax, 0.252, lim(1) + 0.005, 'chance', 'rotation', 90, 'verticalalignment', 'top', 'horizontalalignment', 'left', 'fontsize', 7);
text(ax, lim(2) - 0.01, 0.252, 'chance', 'verticalalignment', 'bottom', 'horizontalalignment', 'right', 'fontsize', 7);
text(ax, bw(2) + 0.005, lim(1) + 0.005, sprintf('within binomial p = 0.05\n(n_{test} = %d-%d)', min(nt_w), max(nt_w)), 'verticalalignment', 'bottom', 'horizontalalignment', 'left', 'fontsize', 7, 'color', [0.35 0.35 0.35]);
text(ax, lim(2) - 0.01, bl(2) + 0.005, sprintf('LOSO binomial p = 0.05 (n_{test} = %d-%d)', min(nt_l), max(nt_l)), 'verticalalignment', 'bottom', 'horizontalalignment', 'right', 'fontsize', 7, 'color', [0.35 0.35 0.35]);
text(ax, lim(2) - 0.01, lim(1) + 0.09, sprintf('%d / %d points below y = x', sum(yl_all < xw_all), numel(xw_all)), 'horizontalalignment', 'right', 'fontsize', 7);
xlabel(ax, 'Within-participant overt accuracy');
ylabel(ax, 'LOSO overt accuracy');
hl = legend(ax, handles, labels, 'location', 'northwest', 'box', 'off', 'fontsize', 7);
set(hl, 'color', 'none');
text(ax, lim(1) + 0.012, 0.405, '(participants above threshold: within | LOSO)', 'fontsize', 6.5, 'color', [0.35 0.35 0.35]);
set(ax, 'layer', 'top');

save_fig(f, S.here, 'fig3_per_subject');
