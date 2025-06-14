function B = svd2eig(A)
% [U, S, V] = svd(A);
% B = U * V';

AA = A' * A;
AA = (AA + AA')/2;
[V, D] = eig(AA); 
D2 = diag(D).^(-.5);
B = A * V * diag(D2) * V';
end