%%%%%%%%%%%%%%%%%%%%%%% First part of Iris task %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Using last 30 samples for training, first 20 for testing %%%%%%%%

%% Constants

n_training = 30;    % Use 30 samples for training
n_testing = 20;     % Use 20 samples for testing
c = 3;              % We have 3 classes
d = 4;              % Dimension of input vectors


%% Load data

setosa_data = load('class_1', '-ascii');
versicolor_data = load('class_2', '-ascii');
virginica_data = load('class_3', '-ascii');

% Use 30 last samples for training, 20 first for testing
set_training = setosa_data(end-n_training+1:end, :);
ver_training = versicolor_data(end-n_training+1:end, :);
vir_training = virginica_data(end-n_training+1:end, :);
training_data = [set_training.', ver_training.', vir_training.'];

set_testing = setosa_data(1:n_testing, :);
ver_testing = versicolor_data(1:n_testing, :);
vir_testing = virginica_data(1:n_testing, :);
testing_data = [set_testing.', ver_testing.', vir_testing.'];


%% Initialize matrices 

W_0 = zeros(C, D);
omega_0 = zeros(C, 1);
W = [W_0, omega_0];
x = zeros(D+1, C*n_training);


%% Training
n_iter = 3000;                  % Number of iterations
alpha = 0.005;                  % Step length
i = 1;
MSE_arr = zeros(n_iter, 1);

while i < n_iter
    grad = 0;
    MSE = 0;
    counter = 1;
    MSE_arr = zeros(n_iter, 1);
    
    for k = 1:C*n_training
        
        t_k = zeros(C, 1);
        if (mod(k, n_training) == 0) && (k ~= C*n_training) 
            counter = counter + 1;
        end
        t_k(counter, :) = 1;
        
        x_k = [training_data(:, k); 1];
        z_k = W*x_k;
        g_k = sigmoid(z_k);
        
        MSE = MSE + .5*(g_k - t_k).'*(g_k - t_k);
        grad = grad + grad_MSE(g_k, t_k, x_k);
    end
    
    MSE_arr(i) = MSE;
    W = W - alpha*grad;
    i = i + 1;
end


%% Testing and confusion matrix

confm_testing = zeros(C, C);
counter = 1;

for k = 1:C*n_testing
    
    x_k = [testing_data(:, k); 1];
    z_k = W*x_k;
    g_k = sigmoid(z_k);
    
    [max_val, i] = max(g_k);
    
    confm_testing(counter, i) = confm_testing(counter, i) + 1;
    
    if (mod(k, n_testing) == 0) && (k ~= C*n_testing)
        counter = counter + 1;
    end
end

% Also create confusion matrix for training samples for comparison
confm_training = zeros(C, C);
counter = 1;

for k = 1:C*n_training
    
    x_k = [training_data(:, k); 1];
    z_k = W*x_k;
    g_k = sigmoid(z_k);
    
    [max_val, i] = max(g_k);
    
    confm_training(counter, i) = confm_training(counter, i) + 1;
    
    if (mod(k, n_training) == 0) && (k ~= C*n_training)
        counter = counter + 1;
    end
end


%% Error rate

err_testing = (C*n_testing - trace(confm_testing))/(C*n_testing);
err_training = (C*n_training - trace(confm_training))/(C*n_training);




