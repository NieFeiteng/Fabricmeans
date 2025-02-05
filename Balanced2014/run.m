file_name = 'balanced_kmeans_resuld.csv';  % 替换为你的 .m 文件名

cluse_num = 5;
avg_time = 10.5;
avg_iter_num = 20.3;
avg_obj_max = 100.7;
standard_deviation_time = 2.1;
standard_deviation_obj = 50.2;
k =1;
% 打开文件以便写入，'w' 表示写入模式
fid = fopen(file_name, 'a');
% fprintf(fid, 'data_name,Clusters num,Average time,iterations,objective function value,Average clusters size\n');

% 写入数据到文件
fprintf(fid, '%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f',file_name_with_ext, cluse_num, avg_time, avg_iter_num, avg_obj_max,standard_deviation_time,standard_deviation_obj);
for ii = 1:k
    fprintf(fid, ',%.2f', avg_cluster_sizes(ii));
end
fprintf(fid, '\n');

% 关闭文件
fclose(fid);