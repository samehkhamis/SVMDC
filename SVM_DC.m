function [w] = SVM_DC(X, y, C, norm, verbose, tmax, tol)
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
Qii = sum(X.^2, 1) + Dii; % linear

for t = 1:tmax
    err = 0;
    for i = 1:n
        g = (w' * X(:, i)) .* y(i) - 1 + Dii * alpha(i); % linear
        if alpha(i) < tol, g = min(g, 0); end
        if alpha(i) > U, g = max(g, 0); end

        if abs(g) > err, err = abs(g); end
        if abs(g) > tol
            alpha_new = min(max(alpha(i) - g ./ Qii(i), 0), U); % linear
            w = w + ((alpha_new - alpha(i)) .* y(i)) * X(:, i); % linear
            alpha(i) = alpha_new; % linear
        end
    end
    if verbose, fprintf('Iter %3d: %.4f\n', t, err); end
    if err < tol, break; end
end
%sc = w' * xt; % score
