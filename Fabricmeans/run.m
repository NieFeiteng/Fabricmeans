clc

methods = {'BABCDKM','BABCDKM-DP','CDKM', 'kmeansPP'};
clusters_sets = {4};
% rho_sets = [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3];
rho_sets = [0.01];
num_runs = 5;
datasets = {'athlete','census', 'Spanish', 'student','pricerunner','cohort','diabeteshealthindicators1','NHANES'};
datasets = {'athlete', 'Spanish','pricerunner','diabeteshealthindicators1'};
% datasets = {'hmda', 'census1990'};
point_max = [10000];
% point_max = [10000,20000,50000,100000,200000,500000,1000000,];
% point_max = [50000,100000,200000,500000,1000000];
% point_max = [500000,1000000,2000000];
threads_set = 4:1:4;
% epsilon_set = [0.5,1,2, 5,8,10];
epsilon_set = [5];
iter_rounds = 100;
% block_sizes = [4,16,64,256,1024];
block_sizes = [16];
seed = [4,5,6,7,8,9,10,11];
% seed = [12,13,14,15,16];

for method_idx = 1:1
    method_name = methods{method_idx};
    result_file_name = ['result_petero_1_', method_name, '.csv'];
    fid = fopen(result_file_name, 'a');

    if ftell(fid) == 0
        fprintf(fid, 'data_name,Clusters num,Average time,iterations,sse,obj,balance loss,Time Std,Obj Std,num-thr,t,Average clusters size\n');
    end
    %   500-1000
    for iter_dataset = 1:length(datasets)
        dataset_name = datasets{iter_dataset};

        for threads_num = threads_set
            threads = threads_num;
            for iter_cluster = 1:length(clusters_sets)
                c = clusters_sets{iter_cluster};
                for point_idx = 1:length(point_max)
                    max_points = point_max(point_idx);
                    for rho_idx = 1:length(rho_sets)
%                         t = rho_sets(rho_idx);
                                t = 0.00005;
                    while t <= 0.4
                            if t < 0.001
                                t = t + 0.00005; 
                            elseif t < 0.01
                                t = t + 0.0005; 
                            elseif t < 0.1
                                t = t + 0.001;  
                            elseif t < 1
                                t = t + 0.01;  
                            end
                        for epsilon_idx = 1:length(epsilon_set)
                            for block_sizes_idx = 1:length(block_sizes)
                                block_size = block_sizes(block_sizes_idx);
                                epsilon = epsilon_set(epsilon_idx);
                                file_path = strcat('\output\', dataset_name, '_', num2str(max_points), '_', num2str(c), '.csv');

                                
                                X = csvread(file_path, 1, 1)';

                                avg_obj_max = 0;
                                avg_sse_max = 0;
                                avg_balance_loss = 0;
                                avg_iter_num = 0;
                                avg_time = 0;
                                avg_cluster_sizes = zeros(1, c);
                                avg_sse = zeros(1, iter_rounds);
                                avg_obj = zeros(1, iter_rounds);

                                time_record = zeros(1, num_runs);
                                loss_record = zeros(1, num_runs);

                                for ite_run = 1:num_runs
                                    fprintf('Running %s on dataset %s, run %d\n', method_name, dataset_name, ite_run);
                                    rng( seed(ite_run));
                                    label = kmeans(X', c);

                                    delete(gcp('nocreate'));

                                    switch method_name
                                        case 'kmeansPP'
                                            [Y_label, sse, elapsed_time] = kmeansPP(X, c);
                                            iter_num = NaN;
                                            obj = sse
                                            balance_loss= 0;
                                        case 'CDKM'
                                            [Y_label, ~, iter_num, sse, elapsed_time] = CDKM(X, label, c, iter_rounds);
                                            obj = sse;
                                            obj(iter_num)
                                        case 'BABCDKM'
                                            rho = 0.01;
                                            [Y_label, ~, iter_num, sse, obj,balance_loss, elapsed_time] = BABCDKM(X, label, c, block_size, t,threads, iter_rounds);
                                        case 'BABCDKM-DP'
                                            rho = 0.01;
                                            [Y_label, ~, iter_num, sse, obj,balance_loss, elapsed_time] = BABCDKM_DP(X, label, c, block_size, t,threads, iter_rounds,epsilon);

                                    end

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
                                    time_record(ite_run) = elapsed_time;
                                    if isnan(iter_num)
                                        loss_record(ite_run) = obj;
                                    else
                                        loss_record(ite_run) = obj(iter_num);
                                    end
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
                                %                             avg_sse = avg_sse / num_runs;
                                avg_balance_loss = avg_balance_loss / num_runs;
                                % Compute standard deviations
                                standard_deviation_time = std(time_record);
                                standard_deviation_obj = std(loss_record);

                                % Prepare the data for output
                                [~, data_name, ext] = fileparts(file_path);
                                file_name_with_ext = strcat(data_name, ext);


                                num_threads = threads;
                                % Write the results to the file
                                fprintf(fid, '%s,%d,%d, %.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,', file_name_with_ext, c,block_size, avg_time, avg_iter_num, avg_sse_max, avg_obj_max,avg_balance_loss,standard_deviation_time, standard_deviation_obj);
                                fprintf(fid, '%.2f,%f', num_threads, t);
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
                            end
                        end
                    end
                    end
                end
            end
        end
    end

    fclose(fid);
end