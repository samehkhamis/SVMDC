function [P, m] = PCA(X, k)
[d, n] = size(X);
if ~exist('k', 'var'), k = min(n - 1, d); end
m = mean(X, 2);
Xm = bsxfun(@minus, X, m);

S = Xm * Xm';
[P, D] = eigs(S, k);
P = P';
%i = find(cumsum(diag(D)) ./ sum(diag(D)) >= e, 1);
