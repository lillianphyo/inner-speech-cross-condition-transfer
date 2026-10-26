function [m, subjects, ntest] = subj_means(T, mask)
  if ~any(mask)
    error('subj_means: empty selection');
  end
  subjects = unique(T.subject(mask));
  m = nan(numel(subjects), 1);
  ntest = nan(numel(subjects), 1);
  for i = 1:numel(subjects)
    sel = mask & strcmp(T.subject, subjects{i});
    m(i) = mean(T.accuracy(sel));
    if isfield(T, 'session')
      key = strcat(T.session(sel), '|', num2str(T.n_test(sel)));
      if numel(unique(key)) ~= numel(unique(T.session(sel)))
        error('subj_means: %s has a session with several n_test values', subjects{i});
      end
      ntest(i) = sum(T.n_test(sel)) / nnz(sel);
    else
      nt = unique(T.n_test(sel));
      if numel(nt) ~= 1
        error('subj_means: %s has %d distinct n_test values', subjects{i}, numel(nt));
      end
      ntest(i) = nt;
    end
  end
end
