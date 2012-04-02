function [Y] = normalize(X)
Y = bsxfun(@rdivide, X, sum(X, 1));
