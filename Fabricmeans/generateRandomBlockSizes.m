function block_sizes = generateRandomBlockSizes(n, block_number, seed)
    if block_number <= 0 || n <= 0 || block_number > n
        error('block_number \n');
    end
    
    rng(seed);

    cut_points = sort(randperm(n - 1, block_number - 1));
    
    cut_points = [0, cut_points, n];
    
    block_sizes = diff(cut_points);
end
