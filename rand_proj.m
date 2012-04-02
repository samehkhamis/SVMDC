function [R] = rand_proj(k, d)
R = randn(k, d);
%R = bsxfun(@rdivide, R, sqrt(sum(R.^2, 1)));
R = R';
