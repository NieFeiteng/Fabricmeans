input_path = 'path\to\data\';
output_path = 'data\';  
if ~exist(output_path, 'dir')
    mkdir(output_path);
end
point_max = [50000,100000,200000,500000,1000000];
l = 10000;

datasets = {'athlete','census', 'Spanish','YearPredictionMSD10000'};

k = 10;


for i = 1:length(datasets)
    dataset_name = datasets{i};
    fprintf('Processing dataset: %s\n', dataset_name);

    file_path = strcat('path\to\data\', dataset_name,'.csv');
    X = csvread(file_path, 1, 1);
    max_points =min(point_max(1), size(X,2));
    l = min(l,size(X,1));
    X = X(1:l,1:max_points);
    X = normalize(X, 1, 'range');

    n = size(X, 1);

    y = kmeans(X, k, 'Replicates', 10);

    rand_idx = randperm(n);
    g = zeros(n, 1);
    half_n = floor(n / 2);
    g(rand_idx(1:half_n)) = 1;
    g(rand_idx(half_n+1:end)) = 2;
    g = g';
    bac_Y = cell(1,1);


    Y = zeros(n, k);
for idx = 1:n
    Y(idx, y(idx)) = 1;
end

    bac_Y{1} = Y;

    output_file = fullfile(output_path, [dataset_name '.mat']);
    save(output_file, 'bac_Y', 'y', 'g', '-v7.3');
end
