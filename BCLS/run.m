%   Balanced Clustering with Leas Square Regression
%   Main function
close all;
clear all;

globals;
datasets = {'athlete','census', 'Spanish', 'student','pricerunner','cohort','diabeteshealthindicators1','NHANES'};

seed = [4,5,6,7,8,9,10,11];
datasets = {'census1990', 'hmda'};

data = 'Balance_EXP';
point_max = [10000];

% point_max = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 2000000];
point_max = [50000, 100000,200000];
seed = [3,4,5,6,7,8,9,10];
for iter_dataset = 1:length(datasets)
    dataset_name = datasets{iter_dataset};

    for point_idx = 1:length(point_max)
        max_points = point_max(point_idx);

%         for cluse_num = 3:1:10
        for cluse_num = 4:1:4

            %% initialization
            display('Initializing data...');
            
            % 初始化存储结果的变量

            ITER = 100;

            num_runs =2;

            time_record = [];
            obj_record = [];

            avg_obj_max = 0;
            avg_iter_num = 0;
            avg_time = 0;
            avg_cluster_sizes = zeros(1, cluse_num);

            for ite_run = 1:num_runs
                seed = ite_run;
                initialization(data, dataset_name, max_points, cluse_num, tmp_dir, infRes, seed);
                load([tmp_dir 'init.mat']);
                [d,n] = size(X);
    

                fprintf('Running : %.2f\n', ite_run);
                fprintf("BLCS!\n");
                StartInd = randsrc(n,1,1:c); Y0 = TransformL(StartInd, c); save([tmp_dir 'Y0'], 'Y0');
                load([tmp_dir 'Y0']);
                run_time = tic;

                %% Optimization
                display('Optimizing...');
                [ID, Y, Obj, Obj2] = BCLS_ALM(X, Y0, ITER, gamma, lam, mu);

                elapsed_time = toc(run_time);

                disp(['Elapsed time: ', num2str(elapsed_time)]);

                %% Evaluation
                ys = sum(Y);
                % result = ClusteringMeasure(gt, ID);
                % ACC = result(1);
                % NMI = result(2);
                % [entropy,~,~] = BalanceEvl(c, ys);

                %% Show the results
                % visualization(data, ID, la, X, n, c);
                % figure; plot(Obj);
                % figure; plot(Obj2);
                % figure; stem(ys);
                % disp(ys);

                avg_obj_max = avg_obj_max + Obj2(ITER);
                avg_iter_num = avg_iter_num + ITER;
                avg_time = avg_time + elapsed_time;
                time_record(ite_run) = elapsed_time;
                obj_record(ite_run) = Obj2(ITER);

                % 计算每个簇的大小
                for ii = 1:c
                    avg_cluster_sizes(ii) = avg_cluster_sizes(ii) + ys(ii);
                end
            end


            avg_obj_max = avg_obj_max / num_runs;
            avg_iter_num = avg_iter_num / num_runs;
            avg_time = avg_time / num_runs;
            avg_cluster_sizes = avg_cluster_sizes / num_runs;


            disp(['Average number of iterations: ', num2str(avg_iter_num)]);
            disp(['Average final objective function value: ', num2str(avg_obj_max)]);
            disp(['Average time: ', num2str(avg_time)]);
            for ii = 1:c
                fprintf('Average cluster %d size: %.2f\n', ii, avg_cluster_sizes(ii));
            end


            variance_time = var(time_record);
            variance_obj = var(obj_record);

            standard_deviation_time = std(time_record);
            standard_deviation_obj = std(obj_record);


            file_name = 'BCLS(AAAI)_result.csv';
            fid = fopen(file_name, 'a');
            %         fprintf(fid, 'data_name,Clusters num,Average time,iterations,objective function value,Time Std,Obj Std,num-thr,Average clusters size\n');


            file_path = strcat("\output\", dataset_name, "_", num2str(max_points), "_", num2str(cluse_num), ".csv");
            
            [~, data_file_name, ext] = fileparts(file_path);
            file_name_with_ext = strcat(data_file_name, ext);

            fprintf(fid, '%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d',file_name_with_ext, cluse_num, avg_time, avg_iter_num, avg_obj_max,standard_deviation_time,standard_deviation_obj,1);
            for ii = 1:c
                fprintf(fid, ',%.2f', avg_cluster_sizes(ii));
            end
            fprintf(fid, '\n');
            fclose(fid);
        end
    end
end



