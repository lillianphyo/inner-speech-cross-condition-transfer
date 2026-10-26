function mask = pick(T, varargin)
  mask = true(T.n, 1);
  for i = 1:2:numel(varargin)
    mask = mask & strcmp(T.(varargin{i}), varargin{i+1});
  end
end
