function [y] = nearest_neighbor(X, mi)
n = size(X, 2);
c = size(mi, 2);

bestd = sum(bsxfun(@minus, X, mi(:, 1)).^2, 1);
y = ones(1, n);
for i = 2:c
	d = sum(bsxfun(@minus, X, mi(:, i)).^2, 1);
	y(d < bestd) = i;
    bestd = min(d, bestd);
end
