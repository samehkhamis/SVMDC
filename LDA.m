function [W, mi] = LDA(X, y, reg)
if ~exist('reg', 'var'), reg = 1e-4; end
c = max(y);
d = size(X, 1);

Sw = zeros(d, d);
Sb = zeros(d, d);
m = mean(X, 2);
mi = zeros(d, c);
for i = 1:c
	yi = (y == i);
	ni = sum(yi);
	mi(:, i) = mean(X(:, yi), 2);
    mim = mi(:, i) - m;
	Xm = bsxfun(@minus, X(:, yi), mi(:, i));
    Sb = Sb + ni .* (mim * mim');
	Sw = Sw + Xm * Xm';
end

Sw = Sw + eye(d) * reg;
[W, D] = eigs(Sb, Sw, min(c - 1, d - 1));
W = W';
