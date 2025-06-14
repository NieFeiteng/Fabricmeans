clear;
clc;
seed_global = 1;
rng( seed_global);

point_max = [50000,100000,200000,500000];
point_max = [10000,20000];
l = 100;

datasets = {'taxi','census','YearPredictionMSD10000', 'loan','diabetes','har'};
datasets = {'census','diabetes','har','loan','patient','student','taxi','YearPredictionMSD10000'};
% datasets = {'letter'};
datasets = {'hmda', 'census1990'};
% datasets = {'census1990'};
clusters_sets = {3,4,5,6,7,8,9,10};
clusters_sets = {4};
header = {'Method', 'Dataset', 'n', 'k', 'ElapsedTime', 'TotalSSE', 'BalanceLoss'};
for i = 1:max(cell2mat(clusters_sets))
    header{end+1} = sprintf('ClusterSize%d', i);
end


firstWrite = true;

csvFileName = 'results_fce.csv';
fileID = fopen(csvFileName, 'a');


for i = 1:length(datasets)
    for idx_point = 1:length(point_max)
        point_x = point_max(idx_point);
        for iter_cluster = 1:length(clusters_sets)
            dataset_name = datasets{i};
            fprintf('Processing dataset: %s\n', dataset_name);

            file_path = strcat('path\to\data\', dataset_name,'.csv');
            X = csvread(file_path, 1, 0);
            max_points =min(point_x, size(X,1));
            l = min(l,size(X,2));
            X = X(1:max_points,1:l);

            n = size(X, 1);
            k = clusters_sets{iter_cluster};

            y = randi([1 k], size(X,1), 1);
            rand_idx = randperm(n);
            g = zeros(n, 1);

            rand_n = randi([1, n-1]);
            g(rand_idx(1:rand_n)) = 1;
            g(rand_idx(rand_n+1:end)) = 2;
            g = g';
            bac_Y = cell(1,1);

            Y = zeros(n, k);
            for idx = 1:n
                Y(idx, y(idx)) = 1;
            end

            bac_Y{1} = Y;


            nKmeans = 1;
            nClustering = 1;
            %nRuns = floor(nKmeans/nClustering);
            nRuns =1;

            Y = cell(1,nClustering);
            c = length(unique(y));
            G = get_dummies(g);

            lambda1 = 0.001;
            lambda2 = 10;


            t1 = 0;
            t2 = 0;


            tic;
            balance_loss = 0;

            for i=1:nRuns
                cluster_size=zeros(1, c);

                for j=1:nClustering
                    Y{j} = bac_Y{nClustering*(i-1) + j};
                end
                [pre_Y] = fce(Y,G,lambda1,lambda2,X);


                % 计算 SSE 并打印每个簇的大小
                [n, c] = size(pre_Y);
                [n, l] = size(X);
                total_sse = 0;

                for k = 1:c
                    cluster_indices = find(pre_Y(:, k) == 1);
                    cluster_size(k) = length(cluster_indices);

                    if cluster_size(k) > 0
                        cluster_points = X(cluster_indices, :);
                        centroid = mean(cluster_points, 1); 

                        diffs = cluster_points - centroid;      
                        cluster_sse = sum(sum(diffs.^2));        
                        total_sse = total_sse + cluster_sse;



                    end
                    balance_loss = balance_loss + (n/c - cluster_size(k))^2;
                end

                sdcs = sqrt(balance_loss / (c - 1));
                elapsedTime = toc;

                methodName = 'fce';

                fprintf(fileID, '%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f', methodName, dataset_name, n, k, elapsedTime, total_sse, sdcs, balance_loss);
                for cs = 1:length(cluster_size)
                    fprintf(fileID, ',%d', cluster_size(cs));
                end
                fprintf(fileID, '\n');

            end
        end
    end
end


