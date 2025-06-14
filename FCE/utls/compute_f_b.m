function [tmp4] = compute_f_b(G,Y)
c = size(Y,2);
D = G'*Y;
tmp1 = sum(D,2);
D2 = D./tmp1;
A1 = c.*D2;
A2 = 1./(max(c.*D2,eps));
tmp2 = min(A1(:));
tmp3 = min(A2(:));
tmp4 = min(tmp2,tmp3);
end

