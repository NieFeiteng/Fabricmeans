
function [Y, minO, iter_num, sse, obj, balance_loss, elapsed_time] = BABCDKM(X, label,c, block_size, rho, numWorkers, max_iters)


fprintf("BABCDKM\n");
[~,n] = size(X);
run_time = tic;

F = sparse(1:n,label,1,n,c,n);
iter_num = 0;
for i=1:n
    XX(i)=X(:,i)'* X(:,i);
end
XF = X*F;
FF=sum(F,1);    % diag(F'*F) ;
FXXF=XF'*XF;    % F'*X'*X*F;

blocks = partitionNumbers(n,block_size);
par_time_t = zeros(max_iters, length(blocks));

b = (n/c) * 0.001;
alpha = n/(c) - b;
beta = n/(c) + b;
limit_total = zeros(c, 1);
thetas = (n / c) * ones(1, c);
lambdas = zeros(1, c);


for iter_t = 1:max_iters
    iter_num = iter_num+1;
    phi=zeros(1,c);
    v1_t=zeros(1,c);
    v2_t=zeros(1,c);
    V1_all = zeros(length(blocks), c);
    V2_all = zeros(length(blocks), c);
    %% Solve F
    for blockid = 1:length(blocks)
        block = blocks{blockid};
        m = label;

        par_time = tic;
        for idx = 1:length(block)    
            i = block(idx);
            for k = 1:c
                if k == m(i,:)
                    V1 = FXXF(k,k)- 2 * X(:,i)'* XF(:,k)+ XX(i);
                    U1 = V1/ (FF(k) -1) - FXXF(k,k) / FF(k);

                    phi1(idx, k) = U1 - (lambdas(k) - rho * (2 * FF(k) + 1 - 2 * thetas(k)));
                    V1_all(idx, k) = V1;  % Store V1
                else
                    V2 =(FXXF(k,k)  + 2 * X(:,i)'* XF(:,k)+ XX(i));
                    U2 = FXXF(k,k)/ FF(k) -  V2 / (FF(k) +1);

                    phi1(idx, k) = U2 - (lambdas(k) - rho * (2 * FF(k) - 1 - 2 * thetas(k)));
                    V2_all(idx, k) = V2;  % Store V2
                end
            end
        end


        v1_t(block,:)=V1_all(1:length(block),:);
        v2_t(block,:)=V2_all(1:length(block),:);
        phi(block,:) = phi1(1:length(block),:);
        [~,label_update] = min(phi,[],2);
        q = find(m(1:block(end))~=label_update)';
        for j = q
            XF(:,label_update(j))=XF(:,label_update(j))+X(:,j);
            XF(:,m(j))=XF(:,m(j))-X(:,j);
            FF(label_update(j))= FF(label_update(j)) +1;
            FF(m(j))= FF(m(j)) -1;
            %              FXXF(m(j), m(j)) = v1_t(block, m(j));
            %              FXXF(label_update(j), label_update(j)) = V2_all(block, label_update(j));
        end
        label(1:block(end),:)=label_update;
        FXXF=XF'*XF;
    end
    for k = 1:c
        thetas(k) = FF(k) - lambdas(k) / (2 * rho);
        thetas(k) = min(max(thetas(k), alpha), beta);
        lambdas(k) = lambdas(k) + rho * (thetas(k) - FF(k));
    end
    
    for ii=1:c
        idxi = label==ii;
        Xi = X(:,idxi);
        m = size(Xi, 2);
        ceni = mean(Xi,2);
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c);
        balance_loss_OBJ(ii) = lambdas(k) * (thetas(k) - FF(k)) + rho * (thetas(k) - FF(k))^2;
        balance_loss_t(ii) = (thetas(k) - FF(k))^2;
    end
    sse(iter_num) = sum(sumd);

    balance_loss = sum(balance_loss_t);
    obj(iter_num) = sse(iter_num) + sum(balance_loss_OBJ);

end

minO=obj(iter_num);
Y=label;
elapsed_time = toc(run_time);
% row_max_time = max(par_time_t, [], 2);  % max(matrix, [], 2) 
% sum_of_max_time = sum(row_max_time);
% total_sum = sum(sum(par_time_t));
% elapsed_time = elapsed_time - total_sum + sum_of_max_time;

disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
% delete(gcp('nocreate'))

cluster_size = zeros(1, c);
for ii = 1:c
    cluster_size(ii) = sum(label == ii);
end

disp(cluster_size);
minO=min(obj);
Y=label;

end
