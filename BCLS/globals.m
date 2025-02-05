
% Directory setting
base_dir = 'D:\BalanceKMeams\DP_KNN\code\BCLS-master\';
tmp_dir = [base_dir 'tmp/'];

% Parameter setting
gamma = 10^(-5); % the value of Gamma should be between 0 and 10^(-10)
lam = 10^(1);
mu = 1;
infRes = 0.90;    % the percentage of information reserved of the data during PCA dimension reduction
% data = 'UMIST';census1990_500000
% data = 'census1990_500000';
% data = 'UMIST';
iter_seed = 1;
save([tmp_dir 'param.mat']);
