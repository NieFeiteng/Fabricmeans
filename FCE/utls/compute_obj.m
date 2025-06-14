function [min] = compute_obj(ai,Y,H,R,Yi,Ri,G,lambda1,lambda2)
b = size(Yi,2);
C = 0;
   for c=1:b
       C = C + ai(c)^2 .* norm(H - Yi{c}*Ri{c},"fro")^2;
   end
   H = double(H);
   R = double(R);
   Y = double(Y);
   G = double(G);
   min = C + lambda1 * norm(H*R - Y,"fro")^2 + lambda2 * norm(G'*Y,"fro")^2;
end

