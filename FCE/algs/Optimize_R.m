function [R] = Optimize_R(H,Y)

B = Y' * H;

[U ,~ ,V] = svd(B);

R = V * U';

end

