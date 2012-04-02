function [w] = SVM_DC_shrinking(X, y, C, norm, verbose, tmax, tol)
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
M = inf; m = -inf;
A = 1:n;
Qii = sum(X.^2, 1) + Dii; % linear

for t = 1:tmax
    err = 0;
    M_new = -inf; m_new = inf;
    Afilt = true(size(A));
    
    for k = 1:numel(A)
        i = A(k);
        g = (w' * X(:, i)) .* y(i) - 1 + Dii * alpha(i); % linear
        
        if alpha(i) < tol
            g = min(g, 0);
            if g > M, Afilt(k) = 0; end
        end
        if alpha(i) > U
            g = max(g, 0);
            if g < m, Afilt(k) = 0; end
        end
        
        M_new = max(M_new, g);
        m_new = min(m_new, g);
        
        if abs(g) > err, err = abs(g); end
        if abs(g) > tol
            alpha_new = min(max(alpha(i) - g ./ Qii(i), 0), U); % linear
            w = w + ((alpha_new - alpha(i)) .* y(i)) * X(:, i); % linear
            alpha(i) = alpha_new; % linear
        end
    end
    if verbose, fprintf('Iter %3d: %.4f\n', t, err); end
    if err < tol, break; end
    
    A = A(Afilt);
    if M_new - m_new < tol
        if numel(A) == n
            break;
        else
            M = inf; m = -inf;
            A = 1:n;
        end
    end

    if M_new <= tol, M = inf; else M = M_new; end
    if m_new >= -tol, m = -inf; else m = m_new; end
end
%sc = w' * xt; % score
