function T = read_runs(run_dir)
  files = sort(glob(fullfile(run_dir, 'sub-*.csv')));
  if isempty(files)
    error('read_runs: no sub-*.csv in %s', run_dir);
  end
  header = {};
  cols = {};
  for i = 1:numel(files)
    txt = fileread(files{i});
    lines = strsplit(strtrim(txt), "\n");
    hdr = strsplit(strtrim(lines{1}), ',', 'collapsedelimiters', false);
    if isempty(header)
      header = hdr;
      cols = repmat({{}}, 1, numel(header));
    elseif ~isequal(hdr, header)
      error('read_runs: header mismatch in %s', files{i});
    end
    for j = 2:numel(lines)
      fields = strsplit(strtrim(lines{j}), ',', 'collapsedelimiters', false);
      if numel(fields) ~= numel(header)
        error('read_runs: %s line %d has %d fields, expected %d', files{i}, j, numel(fields), numel(header));
      end
      for k = 1:numel(header)
        cols{k}{end+1} = fields{k};
      end
    end
  end
  T = struct();
  for k = 1:numel(header)
    name = regexprep(header{k}, '[^A-Za-z0-9_]', '_');
    T.(name) = cols{k}(:);
  end
  T.accuracy = str2double(T.accuracy);
  T.n_test = str2double(T.n_test);
  T.n = numel(T.accuracy);
end
