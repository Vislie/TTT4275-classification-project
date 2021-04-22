run iris_part1_1

%%%%%%%%%%%%%%%%%%%% Dedicated file for plotting %%%%%%%%%%%%%%%%%%%%%%%%%

%% Scatterplots

figure(1); hold on;         % Sepal lengths vs. sepal width
scatter(setosa_data(:, 2), setosa_data(:, 1), 'filled');
scatter(versicolor_data(:, 2), versicolor_data(:, 1), 'filled');
scatter(virginica_data(:, 2), virginica_data(:, 1), 'filled');
legend('Setosa', 'Versicolor', 'Virginica', 'Location', 'northeast');
xlabel('Sepal width');
ylabel('Sepal length'); 

figure(2); hold on;         % Petal length vs. sepal length
scatter(setosa_data(:, 4), setosa_data(:, 3), 'filled');
scatter(versicolor_data(:, 4), versicolor_data(:, 3), 'filled');
scatter(virginica_data(:, 4), virginica_data(:, 3), 'filled');
legend('Setosa', 'Versicolor', 'Virginica', 'Location', 'northwest');
xlabel('Petal width');
ylabel('Petal length'); 


%% Histograms

figure(3); 
subplot(3, 1, 1);        % Sepal lengths
histogram(setosa_data(:, 1), 20);
xlim([4 8]);
subplot(3, 1, 2); 
histogram(versicolor_data(:, 1), 20);
xlim([4 8]);
subplot(3, 1, 3); 
histogram(virginica_data(:, 1), 20);
xlim([4 8]);
sgtitle('Sepal lengths');

figure(4); 
subplot(3, 1, 1);         % Setosa widths
histogram(setosa_data(:, 2), 20);
xlim([1.8 4.5]);
subplot(3, 1, 2);   
histogram(versicolor_data(:, 2), 20);
xlim([1.8 4.5]);
subplot(3, 1, 3);   
histogram(virginica_data(:, 2), 20);
xlim([1.8 4.5]);
sgtitle('Sepal widths');

figure(5); 
subplot(3, 1, 1);           % Petal lengths
histogram(setosa_data(:, 3), 20);
xlim([0.9 7]);
subplot(3, 1, 2);   
histogram(versicolor_data(:, 3), 20);
xlim([0.9 7]);
subplot(3, 1, 3);   
histogram(virginica_data(:, 3), 20);
xlim([0.9 7]);
sgtitle('Petal lengths');

figure(6); 
subplot(3, 1, 1);         % Petal widths
histogram(setosa_data(:, 4), 20);
xlim([0 2.6]);
subplot(3, 1, 2);   
histogram(versicolor_data(:, 4), 20);
xlim([0 2.6]);
subplot(3, 1, 3);   
histogram(virginica_data(:, 4), 20);
xlim([0 2.6]);
sgtitle('Petal widths');