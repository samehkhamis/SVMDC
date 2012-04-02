function [dataeq] = histeq_all(data)
% Histogram equatlize the images
v = size(data, 3);
n = size(data, 4);
dataeq = zeros(size(data, 1) * size(data, 2), v, n);
for i = 1:n
    for j = 1:v
        dataeq(:, j, i) = reshape(histeq(data(:, :, j, i)), [], 1);
    end
end
