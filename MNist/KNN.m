function output = KNN(k, templateData, templateLabels,testData, testLabels, n_classes)

n_test = length(testData(:,1));

nn_dist = zeros(n_test, k);
nn_index = zeros(n_test, k);

euc_dist = pdist2(templateData, testData);
for i = 1:n_test
    for j = 1:k
       [current_nn_dist, current_nn_index] = min(euc_dist(:,i));
       nn_dist(i,j) = current_nn_dist;
       nn_index(i,j) = current_nn_index;
       
       euc_dist(current_nn_index,i) = inf;
    end
end


classified_labels = zeros(n_test, 1);
for i = 1:n_test
    labels_vote = zeros(n_classes,1);
    for j = 1:k
        labels_vote(templateLabels(nn_index(i, j)) + 1) = ...
            labels_vote(templateLabels(nn_index(i, j)) + 1) + 1;
    end
    [~, max_vote_index] = max(labels_vote);
    classified_labels(i) = max_vote_index - 1;
end

confm_classification = confusionmat(testLabels, classified_labels);

err_classification = (n_test - trace(confm_classification))/(n_test);

s.classified_labels = classified_labels;
s.confusion_matrix = confm_classification;
s.error_rate = err_classification;

output = s;
end

