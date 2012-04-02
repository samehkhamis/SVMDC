function [Y] = whiten(X)
Y = bsxfun(@minus, X, mean(X, 1));
Y = bsxfun(@rdivide, Y, std(Y, 0, 1));
