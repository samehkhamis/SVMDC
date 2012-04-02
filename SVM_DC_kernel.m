function [w] = SVM_DC(X, y, C, norm, verbose, tmax, tol) % unoptimized and barely tested!
if ~exist('norm', 'var'), norm = 1; end
if ~exist('verbose', 'var'), verbose = 1; end
if ~exist('tmax', 'var'), tmax = 1000; end
if ~exist('tol', 'var'), tmax = 1e-3; end

c = unique(y);
if numel(c) == 2
    y(y == c(1)) = -1;
    y(y == c(2)) = 1;
else
    error('Only binary classification is supported.');
end

if norm == 2 % L1- vs L2-SVM
	U = 1e5;
	Dii = 0.5 / C;
else
	U = C;
	Dii = 0;
end

[d, n] = size(X);
alpha = zeros(size(y));
w = zeros(d, 1);
Q = (y' * y) .* (X' * X) + eye(n) .* Dii; % kernel

for t = 1:tmax
    err = 0;
    for i = 1:n
        %%%for loop: g = (Q(i, :) * alpha(i)) - 1; % kernel
        if alpha(i) < tol, g = min(g, 0); end
        if alpha(i) > U, g = max(g, 0); end

        if abs(g) > err, err = abs(g); end
        if abs(g) > tol
            alpha(i) = min(max(alpha(i) - g ./ Q(i, i), 0), U); % kernel
        end
    end
    if verbose, fprintf('Iter %3d: %.4f\n', t, err); end
    if err < tol, break; end
end
%sv = alpha > 0; % support vectors
%sc = sum((X(:, sv)' * xt) .* y(sv) .* alpha(sv)); % score
