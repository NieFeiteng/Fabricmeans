function [Ri] = Optimize_Ri(H, Yi)

n = size(Yi, 2);

Ri = cell(1, n);
Ai = cell(1, n);

for i =1:n
    Ai{i} = H' * Yi{i};
end

Ui = cell(1, n);
Vi = cell(1, n);

for i =1:n
    [Ui{i}, ~ , Vi{i}] = svd(Ai{i});
    Ri{i} = Vi{i} * Ui{i}';
end


end










