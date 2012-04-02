%cd 'Z:\classes\Statistical Pattern Recognition\Project';
%cd 'C:\Users\Administrator\Downloads\FR';

%load data;
%vs = [1, 2];
%data = reshape(face, 24, 21, 3, 200);
%clear face;
%rk = 200;
%pk = 100;

fprintf('DATASET: illumination\n');
fprintf('-----------------------\n');
load illumination;
vs = [2, 5, 8, 11, 14, 17];
data = reshape(illum, 48, 40, 21, 68) / 255;
clear illum;
rk = 500;
pk = 200;

% Equalize and flatten the data, then generate the splits
v = size(data, 3);
n = size(data, 4);
data = histeq_all(data);
for vv = vs
    fprintf('EXPERIMENT: using %d out of %d images per subject for training\n', vv, v);
    Xtr = whiten(reshape(data(:, 1:vv, :), [], n * vv));
    ytr = reshape(repmat((1:n)', 1, vv)', [], 1)';
    Xte = whiten(reshape(data(:, vv + 1:end, :), [], n * (v - vv)));
    yte = reshape(repmat((1:n)', 1, (v - vv))', [], 1)';

    experiments(Xtr, ytr, Xte, yte, pk, rk);
end


fprintf('DATASET: pose\n');
fprintf('-----------------------\n');
load pose;
vs = [2, 5, 8, 11];
data = pose / 255;
clear pose;
rk = 500;
pk = 200;

% Equalize and flatten the data, then generate the splits
v = size(data, 3);
n = size(data, 4);
data = histeq_all(data);
for vv = vs
    fprintf('EXPERIMENT: using %d out of %d images per subject for training\n', vv, v);
    Xtr = whiten(reshape(data(:, 1:vv, :), [], n * vv));
    ytr = reshape(repmat((1:n)', 1, vv)', [], 1)';
    Xte = whiten(reshape(data(:, vv + 1:end, :), [], n * (v - vv)));
    yte = reshape(repmat((1:n)', 1, (v - vv))', [], 1)';

    experiments(Xtr, ytr, Xte, yte, pk, rk);
end


%fprintf('DATASET: yale\n');
%fprintf('-----------------------\n');
%vs = [2, 5, 8];
%data = zeros(50, 60, 11, 15);
%for i = 1:size(data, 4)
%    files = dir(sprintf('yale_faces/subject%02d/*.gif', i));
%    for j = 1:size(data, 3)
%        data(:, :, j, i) = imresize(imread(sprintf('yale_faces/subject%02d/%s', i, files(j).name)), [50, 60]) / 255;
%    end
%end
%rk = 500;
%pk = 200;

% Equalize and flatten the data, then generate the splits
%v = size(data, 3);
%n = size(data, 4);
%data = histeq_all(data);
%for vv = vs
%    fprintf('EXPERIMENT: using %d out of %d images per subject for training\n', vv, v);
%    Xtr = whiten(reshape(data(:, 1:vv, :), [], n * vv));
%    ytr = reshape(repmat((1:n)', 1, vv)', [], 1)';
%    Xte = whiten(reshape(data(:, vv + 1:end, :), [], n * (v - vv)));
%    yte = reshape(repmat((1:n)', 1, (v - vv))', [], 1)';

%    experiments(Xtr, ytr, Xte, yte, pk, rk);
%end
