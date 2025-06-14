function [onehot_labels] = get_dummies(labels)

unique_labels = unique(labels);
    num_classes = length(unique_labels);
    onehot_labels = zeros(length(labels), num_classes);
    for a = 1:length(labels)
        class_idx = find(unique_labels == labels(a));
        onehot_labels(a, class_idx) = 1;
    end
end

