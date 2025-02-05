% Balanced k-means clustering algorithm
% Uses Hungarian algorithm in assignment phase
% Mikko Malinen
% University of Eastern Finland
% 2013


clear all;

tic;

% data

%X =load('X:\clustering\datasets\iris.txt');  % 150 points, k=3, d=4
%X = load('X:\clustering\datasets\subsets\iris_subset_50.txt'); % 50 points, k=3, d= 4
%X = load('X:\clustering\datasets\subsets\thyroid_subset_50.txt'); % 50
%points , k= 2, d= 5
%X = load('X:\clustering\datasets\subsets\wine_subset_50.txt'); % 50
%points, k = 3, d = 13
%X = load('X:\clustering\datasets\subsets\breast_subset_50.txt'); % 50
%points, k= 2, d=9
%X = load('X:\clustering\datasets\subsets\s1_subset_150.txt');  % 150 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s2_subset_150.txt');  % 150 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s3_subset_150.txt');  % 150 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s4_subset_150.txt');  % 150 points, k=15, d=2
%
%X = load('X:\clustering\datasets\subsets\s1_subset_50.txt');  % 50 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s2_subset_50.txt');  % 50 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s3_subset_50.txt');  % 50 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s4_subset_50.txt');  % 50 points, k=15, d=2
% X = load('D:\BalanceKMeams\k-means\Code\Balanced2014\s1_subset_500.txt');  % 500 points, k=15, d=2

% load('D:\BalanceKMeams\k-means\Code\Balanced2014\Mpeg7_uni.mat');  % 500 points, k=15, d=2


% X = X';
% load('D:\BalanceKMeams\k-means\Code\Balanced2014\real_sim_half.mat');
% X = fea;


% X = csvread('D:\BalanceKMeams\k-means\Code\Data\census1990100000.csv',1,1)';





% X = load('D:\BalanceKMeams\k-means\Code\Balanced2014\yeast_uni.mat');  % 500 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s2_subset_500.txt');  % 500 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s3_subset_500.txt');  % 500 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s4_subset_500.txt');  % 500 points, k=15, d=2
%X = load('X:\clustering\datasets\subsets\s1_subset_1000.txt');  % 1000 points, k=15, d=2
%X = X(1:200,:);
%X = load('X:\clustering\datasets\s2.txt');  % 5000 points, k=15, d=2
%X = load('X:\clustering\datasets\thyroid.txt'); % 5dim. 215 points 2 clust
%X = load('X:\clustering\datasets\wine.txt'); % 13 dim, 178 points, 3 clust

% rottavalikoitu.txt
%fid = fopen('rottavalikoitu.txt');
% n vectors in 14 dimensions, 1st dim is gender
%X = textscan(fid,'%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f')
%fclose(fid);
%X = cell2mat(X);
%X

% normalizing rottavalikoitu.txt
%for i=1:size(X,2) % dimensions
%    X(:,i) = (X(:,i)-mean(X(:,i)))/std(X(:,i))
%end
num_runs = 5;  


datasets = {'athlete','census', 'Spanish', 'student', 'pricerunner','cohort','diabeteshealthindicators1','NHANES'};


point_max = [10000];
% point_max = [1000, 2000];

for iter_dataset = 1:length(datasets)
    dataset_name = datasets{iter_dataset};

    time_record = [];
    obj_record = [];


    for point_idx = 1:length(point_max)

        max_points = point_max(point_idx);

        for cluse_num = 4:1:4
            fprintf('dataset_name : %s\n', dataset_name);
            file_path =strcat("\path\to\data", dataset_name, "_10000_", num2str(cluse_num), ".csv");

            X = csvread(file_path,1,1);

            [~, data_name, ext] = fileparts(file_path);
            file_name_with_ext = strcat(data_name, ext);

            %     tokens = regexp(file_path, '.*\[(\d+)\]\.csv$', 'tokens');
            %     k = str2double(tokens{1}{1});
            k = cluse_num;

            % number of clusters
            % k = 2;

            avg_obj_max = 0;
            avg_iter_num = 0;
            avg_time = 0;
            avg_cluster_sizes = zeros(1, k);
            % number of points
            [n,d] = size(X);
            C = zeros(n,d);
            for ite_run = 1:num_runs
                fprintf('Running : %.2f\n', ite_run);
                fprintf("Balanced-Kmeans 2014!\n");

                run_time = tic;
                minimum_size_of_a_cluster = floor(n/k);

                % dimensionality
                d = size(X,2);
                MSE_best = 0; % dummy value
                number_of_iterations_distribution = zeros(100,1);

                for repeats = 1:1      % 1:100
                    % initial centroids
                    for j = 1:k
                        pass = 0;
                        while pass == 0
                            %                     rng(ite_run);
                            i = randi(n);
                            pass = 1;
                            for l = 1:j-1
                                if X(i,:) == C(l,:) pass = 0;
                                end
                            end
                        end
                        C(j,:) = X(i,:);
                    end


                    partition = 0;                 % dummy value
                    partition_previous = -1;       % dummy value
                    partition_changed = 1;

                    kmeans_iteration_number = 0;

                    while ((partition_changed)&&(kmeans_iteration_number<100))% kmeans iterations

                        partition_previous = partition;

                        % kmeans assignment step

                        % setting cost matrix for Hungarian algorithm
                        costMat = zeros(n);
                        for i=1:n
                            for j = 1:n
                                costMat(i,j) = (X(j,:)-C(mod(i,k)+1,:))*(X(j,:)-C(mod(i,k)+1,:))';
                            end
                        end

                        % Execute Hungarian algorithm
                        [assignment,cost] = munkres(costMat);

                        % zero partitioning
                        for i = 1:n
                            partition(i) = 0;
                        end

                        % find current partitioning from hungarian algorithm result
                        for i = 1:n
                            if assignment(i) ~= 0
                                partition(assignment(i))=mod(i,k)+1;
                            end
                        end

                        % kmeans update step

                        for j = 1:k
                            C(j,:) = mean(X(find(partition==j),:));
                        end


                        kmeans_iteration_number = kmeans_iteration_number +1;

                        partition_changed = sum(partition~=partition_previous);

                    end  % kmeans iterations


                    MSE = 0;
                    for i = 1:n
                        %     MSE = MSE +
                        %     ((X(i,:)-C(partition(i),:))*(X(i,:)-C(partition(i),:))')/n;   %我们的算法没有除以点数量
                        MSE = MSE + ((X(i,:)-C(partition(i),:))*(X(i,:)-C(partition(i),:))');
                    end

                    if (MSE<MSE_best)||(repeats==1)
                        MSE_best = MSE;
                        C_best = C;
                        partition_best = partition;
                    end

                    MSE_repeats(repeats) = MSE;

                    number_of_iterations_distribution(kmeans_iteration_number) = number_of_iterations_distribution(kmeans_iteration_number)+1;

                end % repeats


                % new notation
                elapsed_time = toc(run_time);
                disp(['Elapsed time: ', num2str(elapsed_time)]);

                C = C_best;
                partition = partition_best;
                MSE = MSE_best;

                %     for j = 1:k
                %         num_points_in_cluster = sum(partition == j);
                %         fprintf('Cluster %d: %d points\n', j, num_points_in_cluster);
                %     end

                avg_obj_max = avg_obj_max + MSE;
                avg_iter_num = avg_iter_num + kmeans_iteration_number;
                avg_time = avg_time + elapsed_time;

                time_record(ite_run) = elapsed_time;
                obj_record(ite_run) = MSE;

                for ii = 1:k
                    avg_cluster_sizes(ii) = avg_cluster_sizes(ii) + sum(partition == j);
                end


                %     number_of_iterations_distribution
                %     MSE
                fprintf('MSE: %.2f\n', MSE);
                %     mean_MSE_repeats = mean(MSE_repeats);
                %     std_MSE_repeats = std(MSE_repeats);

                % figure
                % X(:,1) = X(:,3);  % to view some other dimension
                % plot(C(:,1),C(:,2),'gO');
                % hold on
                % plot(X(find(partition==1),1),X(find(partition==1),2),'r+');
                % if k>1
                %     hold on
                %     plot(X(find(partition==2),1),X(find(partition==2),2),'bO');
                % end
                % if k>2
                %     hold on
                %     plot(X(find(partition==3),1),X(find(partition==3),2),'r.');
                % end
                % if k>3
                %     hold on




                %     plot(X(find(partition==4),1),X(find(partition==4),2),'b.');
                % end
                % if k>4
                %     hold on
                %     plot(X(find(partition==5),1),X(find(partition==5),2),'r+');
                % end
                % if k>5
                %     hold on
                %     plot(X(find(partition==6),1),X(find(partition==6),2),'bO');
                % end
                % if k>6
                %     hold on
                %     plot(X(find(partition==7),1),X(find(partition==7),2),'b+');
                % end
                % if k>7
                %     hold on
                %     plot(X(find(partition==8),1),X(find(partition==8),2),'b+');
                % end
                % if k>8
                %     hold on
                %     plot(X(find(partition==9),1),X(find(partition==9),2),'r.');
                % end
                % if k>9
                %     hold on
                %     plot(X(find(partition==10),1),X(find(partition==10),2),'b.');
                % end
                % if k>10
                %     hold on
                %     plot(X(find(partition==11),1),X(find(partition==11),2),'g+');
                % end
                % if k>11
                %     hold on
                %     plot(X(find(partition==12),1),X(find(partition==12),2),'gO');
                % end
                % if k>12
                %     hold on
                %     plot(X(find(partition==13),1),X(find(partition==13),2),'g+');
                % end
                % if k>13
                %     hold on
                %     plot(X(find(partition==14),1),X(find(partition==14),2),'g.');
                % end
                % if k>14
                %     hold on
                %     plot(X(find(partition==15),1),X(find(partition==15),2),'g.');
                % end
                %
                % toc;

            end

            avg_obj_max = avg_obj_max / num_runs;
            avg_iter_num = avg_iter_num / num_runs;
            avg_time = avg_time / num_runs;
            for ii = 1:k
                avg_cluster_sizes(ii) = avg_cluster_sizes(ii) / num_runs;
            end





            disp(['Average number of iterations: ', num2str(avg_iter_num)]);
            disp(['Average final objective function value: ', num2str(avg_obj_max)]);
            disp(['Average time: ', num2str(avg_time)]);
            for ii = 1:k
                fprintf('Average cluster %d size: %.2f\n', ii, avg_cluster_sizes(ii));
            end

            variance_time = var(time_record);
            variance_obj = var(obj_record);

            standard_deviation_time = std(time_record);
            standard_deviation_obj = std(obj_record);


            file_name = 'balanced_kmeans_resuld.csv'; 


            fid = fopen(file_name, 'a');
            % fprintf(fid, 'data_name,Clusters num,Average time,iterations,objective function value,Average clusters size\n');

            fprintf(fid, '%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f',file_name_with_ext, cluse_num, avg_time, avg_iter_num, avg_obj_max,standard_deviation_time,standard_deviation_obj);
            for ii = 1:k
                fprintf(fid, ',%.2f', avg_cluster_sizes(ii));
            end
            fprintf(fid, '\n');

            fclose(fid);
        end
    end
end

