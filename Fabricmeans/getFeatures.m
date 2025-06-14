function blocks = getFeatures(n, block_number, seed, israndom)
%PARTITIONFEATURES Divides numbers from 1 to n into specified number of blocks
%   with user-defined (uneven) block sizes.
%
%   blocks = partitionFeatures(n, block_number, block_sizes) partitions the
%   integers from 1 to n into a cell array of blocks, where each block has a
%   size defined in the `block_sizes` vector. The sum of `block_sizes` must be equal to n.
%
%   Inputs:
%       n             - Total number of elements (positive integer)
%       block_number  - Desired number of blocks (positive integer)
%       block_sizes   - A vector specifying the size of each block (positive integers)
%
%   Output:
%       blocks        - 1 x block_number cell array, each cell contains a
%                        vector of indices for that block.
if(israndom)
    fprintf("Features random distribution\n");
    block_sizes = generateRandomBlockSizes(n, block_number, seed);



    if ~isscalar(n) || n <= 0 || n ~= floor(n)
        error('n must be a positive integer.');
    end
    if ~isscalar(block_number) || block_number <= 0 || block_number ~= floor(block_number)
        error('block_number must be a positive integer.');
    end
    if length(block_sizes) ~= block_number
        error('block_sizes must have exactly block_number elements.');
    end
    if sum(block_sizes) ~= n
        error('The sum of block_sizes must equal n.');
    end

    % Initialize blocks
    blocks = cell(1, block_number);

    % Assign indices to each block
    start_idx = 1;
    for k = 1:block_number
        current_block_size = block_sizes(k);
        end_idx = start_idx + current_block_size - 1;
        blocks{k} = start_idx:end_idx;

        start_idx = end_idx + 1;
    end

else
    fprintf("Uniform feature distribution\n");
    if ~isscalar(n) || n <= 0 || n ~= floor(n)
        error('n must be a positive integer.');
    end
    if ~isscalar(block_number) || block_number <= 0 || block_number ~= floor(block_number)
        error('block_number must be a positive integer.');
    end

    % Calculate base block size and number of blocks needing +1
    base_size = floor(n / block_number);
    remainder = mod(n, block_number);

    % Initialize blocks
    blocks = cell(1, block_number);

    % Assign indices to each block
    start_idx = 1;
    for k = 1:block_number
        % Determine block size (first 'remainder' blocks get +1 element)
        current_block_size = base_size + (k <= remainder);

        end_idx = start_idx + current_block_size - 1;
        blocks{k} = start_idx:end_idx;

        start_idx = end_idx + 1;
    end
end
    fprintf('特征分布:');
    for i = 1:length(blocks)
        fprintf('  %d   ', length(blocks{i}));
    end
    fprintf('\n');
%
% function blocks = getFeatures(n, block_number, seed)
%     %PARTITIONFEATURES Divides numbers from 1 to n into specified number of blocks.
%     %
%     %   blocks = partitionFeatures(n, block_number) partitions the integers
%     %   from 1 to n into a cell array of approximately equal sized blocks.
%     %
%     %   Inputs:
%     %       n            - Total number of elements (positive integer)
%     %       block_number - Desired number of blocks (positive integer)
%     %
%     %   Output:
%     %       blocks       - 1 x block_number cell array, each cell contains
%     %                      a vector of indices for that block.
%
%     if ~isscalar(n) || n <= 0 || n ~= floor(n)
%         error('n must be a positive integer.');
%     end
%     if ~isscalar(block_number) || block_number <= 0 || block_number ~= floor(block_number)
%         error('block_number must be a positive integer.');
%     end
%
%     % Calculate base block size and number of blocks needing +1
%     base_size = floor(n / block_number);
%     remainder = mod(n, block_number);
%
%     % Initialize blocks
%     blocks = cell(1, block_number);
%
%     % Assign indices to each block
%     start_idx = 1;
%     for k = 1:block_number
%         % Determine block size (first 'remainder' blocks get +1 element)
%         current_block_size = base_size + (k <= remainder);
%
%         end_idx = start_idx + current_block_size - 1;
%         blocks{k} = start_idx:end_idx;
%
%         start_idx = end_idx + 1;
%     end
% end