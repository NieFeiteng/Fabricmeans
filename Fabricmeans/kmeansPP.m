
function [Y, obj, elapsed_time] = kmeansPP(X, c)
fprintf("Kmeans++!\n");
run_time = tic;
Y = kmeans(X', c);

elapsed_time = toc(run_time);

disp(['Elapsed time: ', num2str(elapsed_time)]);

for ii=1:c
    idxi = find(Y==ii);
    Xi = X(:,idxi);
    ceni = mean(Xi,2);
    center1(:,ii) = ceni;
    c2 = ceni'*ceni;
    d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
    sumd(ii,1) = sum(d2c);
end

obj = sum(sumd) ;     %  objective function value