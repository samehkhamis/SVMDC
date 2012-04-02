function [i] = argmax(X)
[d, i] = max(X, [], 1);
