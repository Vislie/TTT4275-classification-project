%%%%%%%%%%%%%%%%%%%%%%% First part of MNIST task %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Using first 60 000 samples for training, 10 000 for testing %%%%%%

%% Constants

sz_chunks = 1000;
n_chunks = num_train / sz_chunks;
C = 10;


%% Initialize matrices

nn_data = ones(2*num_test, n_chunks);

%% Find nearest neighbour

for i = 1:n_chunks
    
    train_chunk = trainv((i-1)*sz_chunks+1:i*sz_chunks,:);
    euc_dist = dist(train_chunk, testv.');
    for j = 1:num_test
       [local_nn_dist, local_nn_index] = min(euc_dist(:,j));
       nn_data(j, i) = local_nn_dist;
       nn_data(num_test + j,i) = (i-1)*sz_chunks + local_nn_index;
    end
    fprintf('Chunk %d done\n', i);
end

%% Classify test data based on NN

classified_labels = zeros(num_test, 1);
for i = 1:num_test
   [global_nn_dist, global_nn_index] = min(nn_data(i,:));
   label = trainlab(nn_data(num_test + i, global_nn_index));
   classified_labels(i) = label;
end

confm_classification = confusionmat(testlab, classified_labels);

err_classification = (num_test - trace(confm_classification))/(num_test);

