clc

methods = {'Lloyd','CDKM','Fabricmeans','Fabricmeans-DP'};
clusters_sets = {3,4,5,6,7,8,9,10};

% rho_sets = [0.0001,0.0002,0.0005,0.001,0.005,0.01,0.02,0.05,0.1];
rho_sets = [0.01];
num_runs = 10;
seed_global =  1;

% datasets = { 'census','diabetes','letter','loan','patient','student','taxi'};
datasets = {'hmda', 'census1990'};
israndom = 0;
point_max = [10000,20000,50000,100000,200000,500000,1000000,2000000,5000000];
threads_set = 10;
epsilon_set = [0.5,1,2, 5,8,10];
max_iter = 100;
% block_sizes = [4,16,64,256,1024];
block_sizes = [256];
cvs = [0];
% threads_set = [2,3,4,5,6,7,8,9,10];
threads_set = [10];
% seed = [1,2,3,4,5,10,11,12,13,14,15,16,17,18,19,20];
seed = [1];
for cv_idx = 1:length(cvs)
    israndom = cvs(cv_idx);
    for method_idx = 7:7
        method_name = methods{method_idx};
        %         result_file_name = ['result_big_data', method_name, '.csv'];
        %             result_file_name = ['result_var_rho_t_4_epsilon_5_time.csv'];
        result_file_name = ['result_big_data_time.csv'];
        fid = fopen(result_file_name, 'a');

        if ftell(fid) == 0
            fprintf(fid, 'method_name,data_name,seed,cv,Clusters num,point size,block size,Average time,iterations,sse,obj,balance loss,num-thr,t,epsilon,Average clusters size\n');
        end

        for iter_dataset = 1:length(datasets)
            l = 100;
            dataset_name = datasets{iter_dataset};
            file_path = strcat('path\to\data\', dataset_name,'.csv');
            X_ori = csvread(file_path, 1, 0)';
            for idx_thread = 1:length(threads_set)

                idx_seed = 1
                threads = threads_set(idx_thread);

                for iter_cluster = 1:length(clusters_sets)
                    c = clusters_sets{iter_cluster};
                    for point_idx = 1:length(point_max)
                        max_points =min(point_max(point_idx), size(X_ori,2))  ;
                        l = min(l,size(X_ori,1));
                        X = X_ori(1:l,1:max_points);

                        for rho_idx = 1:length(rho_sets)
                            t = rho_sets(rho_idx);
                            for epsilon_idx = 1:length(epsilon_set)
                                for block_sizes_idx = 1:length(block_sizes)
                                    block_size = block_sizes(block_sizes_idx);
                                    epsilon = epsilon_set(epsilon_idx);
                                    avg_obj_max = 0;
                                    avg_sse_max = 0;
                                    avg_sdcs = 0;
                                    avg_balance_loss = 0;
                                    avg_iter_num = 0;
                                    avg_time = 0;
                                    avg_cluster_sizes = zeros(1, c);
                                    avg_sse = zeros(1, max_iter);
                                    avg_obj = zeros(1, max_iter);
                                    balance_loss_t  = zeros(1, c);

                                    for ite_run = 1:num_runs
                                        fprintf('Running %s on dataset %s, run %d\n', method_name, dataset_name, ite_run);
                                        seed_global = seed(idx_seed);
                                        // label = kmeans(X', c);
                                        label = randi([1 c], size(X,2), 1);
                                        switch method_name
                                            case 'CDKM'
                                                [Y_label, ~, iter_num, sse, elapsed_time] = CDKM(X, label, c, max_iter);
                                                obj = sse;
                                                obj(iter_num)
                                            case 'Fabricmeans'
                                                % block_size =  16;
                                                rho = 0.01;
                                                [Y_label, ~, iter_num, sse, obj,balance_loss, elapsed_time] = Fabricmeans (X, label, c, block_size, t,threads, max_iter);
                                            case 'Lloyd'
                                                [Y_label, ~, iter_num, sse, balance_loss, elapsed_time, size0] = Lloyd(X, label, c, max_iter);
                                                obj = sse;
                                            case 'Fabricmeans-DP'
                                                % block_size =  16;
                                                rho = 0.01;
                                                [Y_label, ~, iter_num, sse, obj,balance_loss, elapsed_time] = Fabricmeans _DP(X, label, c, block_size, t,threads, max_iter,epsilon);

                                        end

                                        for ii = 1:c
                                            balance_loss_t(ii) = (sum(Y_label == ii) - max_points/c)^2;
                                        end
                                        balance_loss = sum(balance_loss_t);
                                        sdcs = sqrt(balance_loss / (c - 1))
                                        if isnan(iter_num)
                                            avg_sse_max = avg_sse_max + sse;
                                            avg_obj_max = avg_obj_max + obj;
                                        else
                                            avg_sse_max = avg_sse_max + sse(iter_num);
                                            avg_obj_max = avg_obj_max + obj(iter_num);
                                        end
                                        avg_iter_num = avg_iter_num + iter_num;
                                        avg_time = avg_time + elapsed_time;
                                        avg_balance_loss = avg_balance_loss + balance_loss;
                                        avg_sdcs = avg_sdcs + sdcs;
                                        time_record(ite_run) = elapsed_time;

                                        for ii = 1:c
                                            avg_cluster_sizes(ii) = avg_cluster_sizes(ii) + sum(Y_label == ii);
                                        end

                                        if ~isnan(iter_num)
                                            avg_obj = avg_obj + obj;
                                        end
                                    end

                                    % Compute averages
                                    avg_obj_max = avg_obj_max / num_runs;
                                    avg_sse_max = avg_sse_max / num_runs;
                                    avg_iter_num = avg_iter_num / num_runs;
                                    avg_time = avg_time / num_runs;
                                    avg_cluster_sizes = avg_cluster_sizes / num_runs;
                                    avg_obj = avg_obj / num_runs;
                                    avg_sdcs = avg_sdcs / num_runs;
                                    %                             avg_sse = avg_sse / num_runs;
                                    avg_balance_loss = avg_balance_loss / num_runs;


                                    % Prepare the data for output
                                    [~, data_name, ext] = fileparts(file_path);
                                    file_name_with_ext = strcat(data_name, ext);
                                    num_threads = threads;


                                    fprintf(fid, '%s,%s,%d,%.2f,%d,%d,%d, %.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,',method_name, file_name_with_ext,seed_global,israndom,max_points, c,block_size, avg_time, avg_iter_num, avg_sse_max, avg_obj_max,avg_sdcs,avg_balance_loss );
                                    fprintf(fid, '%.2f,%f', num_threads, t);
                                    fprintf(fid, ',%f', epsilon);

                                    for ii = 1:c
                                        fprintf(fid, ',%.2f', avg_cluster_sizes(ii));
                                    end
                                    if ~isnan(obj)
                                        fprintf(fid, ',obj');
                                        for ii = 1:length(avg_obj)
                                            if avg_obj(ii) == 0
                                                break;
                                            end
                                            fprintf(fid, ',%.2f', avg_obj(ii));
                                        end
                                    end
                                    fprintf(fid, '\n');

                                    if ~exist(method_name, 'dir') 
                                        mkdir(method_name); 
                                    end
                                    figure('Visible', 'off');
                                    plot(avg_obj);
                                    file_name = sprintf('%s_%s_%.4f_obj_%d.png', method_name, data_name, t, c);
                                    saveas(gcf, fullfile(method_name, file_name));
                                end
                            end
                            %                     end
                        end
                    end
                end
            end
        end
    end
    fclose(fid);
end