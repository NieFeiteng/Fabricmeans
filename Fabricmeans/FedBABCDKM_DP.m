function [Y, minO, iter_num, sse, obj, balance_loss, elapsed_time] = FedBABCDKM_DP(X, label,c, block_size, rho, numWorkers, max_iters, seed,israndom,epsilon)
fprintf("FedBABCDKM-DP\n");
[~,n] = size(X);

clip_param = 0.5;
delta_dp = 0.001;
sigma = sqrt(2 * log(1.25/delta_dp)) * clip_param / epsilon;

for ii=1:c
    idxi = find(label==ii);
    Xi = X(:,idxi);
    ceni = mean(Xi,2);
    center(:,ii) = ceni;
    c2 = ceni'*ceni;
    d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
    sumd(ii,1) = sum(d2c);
end
clients_num = numWorkers;

run_time = tic;
[features_num,n] = size(X);
F = sparse(1:n,label,1,n,c,n);
iter_num = 0;


blocks = partitionNumbers(n, block_size); % partition blocks;
features_clients = getFeatures(features_num, clients_num, seed,israndom); % partition features;

Xm_cell = cell(1, clients_num);
XmXm_cell = cell(1, clients_num);
XmF_cell = cell(1, clients_num);
FXmXmF_cell = cell(1, clients_num);
FF = sum(F,1);    % diag(F'*F) ;

for i = 1:clients_num
    Xm = X(features_clients{i}(1):features_clients{i}(end),:);
    Xm_cell{i} = Xm;
    XmXm = zeros(1,n);
    for j = 1:n
        XmXm(j) = Xm(:,j)'* Xm(:,j);
    end
    XmXm_cell{i} = XmXm;
    XmF = Xm*F;
    XmF_cell{i} = XmF;
    FXmXmF_cell{i} = XmF'*XmF;
end

b = (n/c) * 0.001;
alpha = n/(c) - b;
beta = n/(c) + b;
limit_total = zeros(c, 1);
thetas = (n / c) * ones(1, c);
lambdas = zeros(1, c);

% elapsed_time1 = toc(run_time);
% run_time = tic;
elapsed_time = 0;
for iter_t = 1:max_iters
    iter_start_time = tic;

    iter_num = iter_num+1;
    phi1_t = zeros(block_size, c);
    phi1 = zeros(block_size, c, clients_num);
    phi=zeros(1,c);

    %% Solve F

    for blockid = 1:length(blocks)
        block = blocks{blockid};
        m = label;



        for clientid = 1:clients_num
            Xm = Xm_cell{clientid};
            XmXm = XmXm_cell{clientid};
            XmF = XmF_cell{clientid};
            FXmXmF = FXmXmF_cell{clientid};
            client_start_time = tic;
            for idx = 1:length(block)
                i = block(idx);
                for k = 1:c

                    if k == m(i,:)
                        V1 = FXmXmF(k,k)- 2 * Xm(:,i)'* XmF(:,k)+ XmXm(i);
                        U1 = V1/ (FF(k) -1) - FXmXmF(k,k) / FF(k);
                        phi1(idx, k, clientid) = U1 - ((lambdas(k) - rho * (2 * FF(k) + 1 - 2 * thetas(k)))/clients_num);
                    
                        norm1 = max(phi1(idx, k, clientid), clip_param);
                        noise = sigma * randn(1,1);
                        phi1(idx, k, clientid) = norm1 + noise;
                    
                    else
                        V2 =(FXmXmF(k,k)  + 2 * Xm(:,i)'* XmF(:,k)+ XmXm(i));
                        U2 = FXmXmF(k,k)/ FF(k) -  V2 / (FF(k) +1);
                        phi1(idx, k, clientid) = U2 - ((lambdas(k) - rho * (2 * FF(k) - 1 - 2 * thetas(k)))/clients_num);
                        norm1 = max(phi1(idx, k, clientid), clip_param);
                        noise = sigma * randn(1,1);
                        phi1(idx, k, clientid) = norm1 + noise;
                    end

                end
            end
            client_times(clientid) = toc(client_start_time);
        end
        iter_block_max_time(blockid) = max(client_times);
        iter_block_sum_time(blockid) = sum(client_times);
        phi1_t(:,:) = sum(phi1(:,:,:), 3);

        phi(block,:) = phi1_t(1:length(block),:);
        [~,label_update] = min(phi,[],2);
        q = find(m(1:block(end))~=label_update)';


        for j = q

            FF(label_update(j))= FF(label_update(j)) +1;
            FF(m(j))= FF(m(j)) -1;


            for clientid = 1:clients_num
                Xm = Xm_cell{clientid};
                XmF = XmF_cell{clientid};
                XmF(:,label_update(j))=XmF(:,label_update(j))+Xm(:,j);
                XmF(:,m(j))=XmF(:,m(j))-Xm(:,j);
                XmF_cell{clientid} = XmF;
                FXmXmF_cell{clientid} = XmF'*XmF;
            end
        end
        label(1:block(end),:)=label_update;
        %         F = sparse(1:n,label,1,n,c,n);
    end
    for k = 1:c
        thetas(k) = FF(k) - lambdas(k) / (2 * rho);
        thetas(k) = min(max(thetas(k), alpha), beta);
        lambdas(k) = lambdas(k) + rho * (thetas(k) - FF(k));
    end

    iter_elapsed_time = toc(iter_start_time);
    iter_elapsed_time_max_client = sum(iter_block_max_time);
    iter_federated_time = iter_elapsed_time + iter_elapsed_time_max_client - sum(iter_block_sum_time);
    iter_federated_time = iter_elapsed_time + iter_elapsed_time_max_client - sum(iter_block_sum_time);
    elapsed_time = elapsed_time + iter_federated_time;

    %% compute objective function value
    for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);
        ceni = mean(Xi,2);
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c);
        balance_loss_OBJ(ii) = lambdas(ii) * (thetas(ii) - FF(ii)) + rho * (thetas(ii) - FF(ii))^2;
        balance_loss_t(ii) = (thetas(ii) - FF(ii))^2;
    end
    sse(iter_num) = sum(sumd);


    balance_loss = sum(balance_loss_t);
    obj(iter_num) = sse(iter_num) + sum(balance_loss_OBJ);
end

minO=obj(iter_num);
Y=label;
fprintf('obj(%d)=%f  balance_loss=%f    balance_loss_OBJ=%f  \n   ', iter_num, full(sse(iter_num)), balance_loss,  full(sum(balance_loss_OBJ)));

cluster_size = zeros(1, c);
for ii = 1:c
    cluster_size(ii) = sum(label == ii);
end

disp(cluster_size);
minO=min(obj);
Y=label;

end
