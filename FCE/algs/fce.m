function [Y] = fce(Yi,G,lambda1, lambda2,X)
[nSmp, nCluster] = size(Yi{1});
nBase = size(Yi,2);

Y=zeros(nSmp, nCluster);
for i=1:nSmp
    idxr=randi(nCluster);
    Y(i,idxr)=1;
end
R  = eye(nCluster);
Ri = cell(1,nBase);
for i=1:nBase
    Ri{i} = eye(nCluster);
end
A = zeros(nSmp,nCluster);
for i=1:nBase
    A = A + Yi{i};
end
[U,~,V] = svd(A,'econ');
H = U * V';
ai = ones(1,nBase)./ nBase;

L = lambda1 * eye(nSmp) + lambda2 * (G * G');


maxiter = 20;
for i=1:maxiter
    %*******************************************
    % Update Y
    %*******************************************

    HR = - 2 * lambda1 * H * R;
    [Y, objHistory] = update_Y_fast(L, HR, Y);

    %*******************************************
    % Update Rs
    %*******************************************

    Ri = Optimize_Ri(H, Yi);
    
    %*******************************************
    % Update R
    %*******************************************

    R = Optimize_R(H,Y);
    
    %*******************************************
    % Update H
    %*******************************************

    YRs = zeros(nSmp, nCluster);
    for iBase = 1:nBase
        YRs =  Yi{iBase} * (Ri{iBase} * ai(iBase)^2) + YRs;
    end
    YRs = Y * (R * lambda1)' + YRs;
    H = svd2eig(YRs);
    
    %*******************************************
    % Update ai
    %*******************************************

    ai = Optimize_ai(H, Yi, Ri);

    min = compute_obj(ai,Y,H,R,Yi,Ri,G,lambda1,lambda2);

end
end

