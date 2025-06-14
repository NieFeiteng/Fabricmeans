function [blocks, actual_cv] = getFeatureDistribution(n, T, target_cv, seed)
    if nargin < 4
        seed = 42;
    end
    seed = 42;
    rng(seed);
    tol = 0.05;
    max_iter = 1000;
    mu = n / T;

    lambda_low = 0;
    lambda_high = 10;
    best_raw = [];
    actual_cv = 0;

    for iter = 1:max_iter
        lambda_mid = (lambda_low + lambda_high) / 2;


        r = 2 * rand(1, T) - 1;  % [-1,1]
        x = mu * (1 + lambda_mid * r);


        x(x < 1) = 1;
        x = round(x);
        diff = sum(x) - n;

        while diff ~= 0
            idx = randi(T);
            if diff > 0 && x(idx) > 1
                x(idx) = x(idx) - 1;
                diff = diff - 1;
            elseif diff < 0
                x(idx) = x(idx) + 1;
                diff = diff + 1;
            end
        end


        cur_cv = std(x) / mean(x);

        if abs(cur_cv - target_cv) < tol
            best_raw = x;
            actual_cv = cur_cv;
            break;
        elseif cur_cv > target_cv
            lambda_high = lambda_mid;
        else
            lambda_low = lambda_mid;
        end

        best_raw = x;
        actual_cv = cur_cv;
    end


    blocks = cell(1, T);
    cur_idx = 1;
    for i = 1:T
        count = best_raw(i);
        blocks{i} = cur_idx:(cur_idx + count - 1);
        cur_idx = cur_idx + count;
    end

end
