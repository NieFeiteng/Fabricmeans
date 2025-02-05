function initialization(data, dataset_name, mat_points,cluse_num, tmp_dir, infRes, seed)
%INITIALIZATION Initialize the original data and other variates
%   Detailed explanation goes here

% load data
[Data_ori, gt, c] = loadData(data, dataset_name, mat_points, cluse_num, seed);
[~,n]=size(Data_ori);
% [X, k, share] = pcaInit(Data_ori, infRes);
X = Data_ori;

H = eye(n) - 1/n*ones(n);
X = X*H;                 

% meanX = mean(X);
% X = X - meanX;


save([tmp_dir 'init.mat'], 'X', 'gt', 'c');

end
