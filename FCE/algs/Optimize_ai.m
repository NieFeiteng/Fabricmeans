function [ai] = Optimize_ai(H, Yi, Ri)

b = size(Yi,2);
D = zeros(b, 1);

for i = 1:b
    D(i) = norm(H - Yi{i}*Ri{i}, 'fro')^2;
end

ai = 1 ./ max(D,eps);
ai= ai ./ max(sum(ai),eps);

end

