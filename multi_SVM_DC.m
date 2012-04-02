function [Q] = multi_SVM_DC(X, y, C, norm, verbose, tmax, tol)
if ~exist('norm', 'var'), norm = 1; end
if ~exist('verbose', 'var'), verbose = 1; end
if ~exist('tmax', 'var'), tmax = 1000; end
if ~exist('tol', 'var'), tmax = 1e-3; end

c = unique(y);
[d, n] = size(X);
yb = zeros(size(y));
Q = zeros(numel(c), d);

for i = 1:numel(c)
    yb(:) = -1;
    yb(y == i) = 1;
    Q(i, :) = SVM_DCA(X, yb, C, norm, verbose, tmax, tol);
end
