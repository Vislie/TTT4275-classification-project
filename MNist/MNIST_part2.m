%%%%%%%%%%%%%%%%%%%%%%% Second part of MNIST task %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Using clustering to decrease set of templates %%%%%%%%%%%%%

%% Constants

M = 64;
n_classes = 10;
load('data_all.mat');

%% Initialize matrices and arrays

C = zeros(M*n_classes, vec_size);
nn_data = zeros(num_test, 2);

trainset = [trainv trainlab;];
trainset_sorted = sortrows(trainset, vec_size+1);

trainset_label_idexes = zeros(n_classes,1);
trainset_label_idexes(1) = 1;
num = 0;
for i = 1:num_train
   if trainset_sorted(i, vec_size+1) > num
       trainset_label_idexes(num+2) = i;
       num = num+1;
   end
end
trainset_label_idexes(n_classes + 1) = num_train+1;

trainset_sorted = trainset_sorted(:,1:vec_size);

C_indexes = zeros(M*n_classes, 1);


%% Clustering

for i = 1:n_classes
    trainset_indices =  trainset_label_idexes(i):trainset_label_idexes(i+1)-1;
    [idxi, Ci] = kmeans(trainset_sorted(trainset_indices,:), M);
    C_indices = (i-1)*M+1:i*M;
    C(C_indices,:) = Ci;
    fprintf('Cluster of class %d done\n', i-1);
    C_indexes(C_indices,:) = i-1;
end


%% Find nearest neighbour % Classify test data based on KNN

knn1 = KNN(1, C, C_indexes, testv, testlab, n_classes);
knn7 = KNN(7, C, C_indexes, testv, testlab, n_classes);

