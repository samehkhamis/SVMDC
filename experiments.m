function [] = experiments(Xtr, ytr, Xte, yte, pk, rk)
% LDA
[W, mi] = LDA(Xtr, ytr);
yy = nearest_neighbor(W * Xte, W * mi);
fprintf('LDA: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);

% LDA then SVM
XW = W * Xtr;
Q = multi_SVM_DC(XW, ytr, 10, 0);
XWte = W * Xte;
yy = argmax(Q * XWte);
fprintf('LDA/SVM: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);

% SVM
Q = multi_SVM_DC(Xtr, ytr, 10, 0);
yy = argmax(Q * Xte);
fprintf('SVM: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);

% Random projection then SVM
R = rand_proj(size(Xtr, 1), rk);
XR = R * Xtr;
Q = multi_SVM_DC(XR, ytr, 10, 0);
XRte = R * Xte;
yy = argmax(Q * XRte);
fprintf('RND/SVM: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);

% Random projection then LDA
[W, mi] = LDA(XR, ytr);
yy = nearest_neighbor(W * XRte, W * mi);
fprintf('RND/LDA: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);

% PCA then SVM
[P, m] = PCA(Xtr, pk);
XP = P * bsxfun(@minus, Xtr, m);
Q = multi_SVM_DC(XP, ytr, 10, 0);
XPte = P * bsxfun(@minus, Xte, m);
yy = argmax(Q * XPte);
fprintf('PCA/SVM: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);

% PCA then LDA
[W, mi] = LDA(XP, ytr);
yy = nearest_neighbor(W * XPte, W * mi);
fprintf('PCA/LDA: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);

% Random projection then LDA then SVM
[W, mi] = LDA(XR, ytr);
XW = W * XR;
Q = multi_SVM_DC(XW, ytr, 10, 0);
XWte = W * XRte;
yy = argmax(Q * XWte);
fprintf('RND/LDA/SVM: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);

% PCA then LDA then SVM
[W, mi] = LDA(XP, ytr);
XW = W * XP;
Q = multi_SVM_DC(XW, ytr, 10, 0);
XWte = W * XPte;
yy = argmax(Q * XWte);
fprintf('PCA/LDA/SVM: %.2f%%\n', sum(yy == yte) / numel(yte) * 100);
