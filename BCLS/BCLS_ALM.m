%
function [ID, Y, Obj, Obj2] = BCLS_ALM(X, Y, ITER, gamma, lam, mu)
% BCLS_ALM
% min_Y,W,b ||X'W+1b'-Y||^2 + gamma*||W||^2 + lam*Tr(Z'11'Z) + mu/2*||Y-Z + 1/mu*Lambda||^2
% INPUT:
% X: data matrix (d by n), already processed by PCA with 80%~90% information preserved
% Y: randomly initialized label matrix (n by c)
% Parameters: gamma and lam are the parameters respectively corresponding to Eq.(13) in the paper
% OUTPUT:
% ID: indicator vector (n by 1)
% Y: generated label matrix (b by c)


% ITER = 1200;
[dim, n] = size(X);

H = eye(n) - 1/n*ones(n);
X = X*H;
% meanX = mean(X);

% X = X - meanX;

c = size(Y,2);   % number of clusters
Lambda = zeros(n,c);
rho = 1.005;
P = eye(dim)/(X*X'+gamma*eye(dim));

Obj2 = zeros(ITER, 1);

for iter = 1:ITER

    display(['Solving alternatively...',num2str(iter)]);

    % Solve W and b
    W = P*(X*Y);
    b = mean(Y)';
    E = X'*W + ones(n,1)*b' - Y;

    % Solve Z
    %     Z = (mu*eye(n)+2*lam*ones(n))\(mu*Y + Lambda);   % original solution - O(n^3)
        Z = (-2*lam*ones(n)+(mu+2*n*lam)*eye(n))/(mu^2+2*n*lam*mu)*(mu*Y+Lambda);  % new solution - O(n^2)



    % Solve Y
    V = 1/(2+mu)*(2*X'*W + 2*ones(n,1)*b' + mu*Z - Lambda);
    [~, ind] = max(V,[],2);
    Y = zeros(n,c);
    Y((1:n)' + n*(ind-1)) = 1;

    % Update Lambda and mu according to ALM
    Lambda = Lambda + mu*(Y-Z);
    mu = min(mu*rho, 10^5);

    % Objective value
    sum_Y = sum(Y(:)); 
    Obj(iter) = trace(E'*E) + gamma*trace(W'*W) + lam*(sum_Y^2);


    %     Obj(iter) = trace(E'*E) + gamma*trace(W'*W) + lam*trace(Y'*ones(n)*Y);


    % Calculate Obj2: sum of squared distances from points to cluster centers
    cluster_centers = (X * Y) ./ sum(Y);
    distances = zeros(n, 1);
    for i = 1:n
        distances(i) = norm(X(:, i) - cluster_centers(:, ind(i)))^2;
    end
    Obj2(iter) = sum(distances);


end;

[~,ID] = max(Y,[],2);

end



