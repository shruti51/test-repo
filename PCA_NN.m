%% This function presents a NN which is used to classify 6 sttic hand gestures

tic;

%% Data Input
clear all;
close all;

 %load('C:\Users\Varun Kumar\Dropbox\Codes\Thesis\gesture_autoencoder7classes\TestTrainUnlabelledData.mat');
 load('C:\My Stuff\Thesis\PCA_NN\train_test_data.mat');
 
numClasses = 20; % per class
no_train = 450; % per class
no_test = 110 ; % per class
total_samp = 560; % per class
no_Train_samples = no_train*numClasses;
no_Test_samples = no_test*numClasses;


ind_train = randperm(length(trainLabels));
ind_test = randperm(length(testLabels));

trainData = trainData(:,ind_train);
testData = testData(:,ind_test);
trainLabels = trainLabels(ind_train);
testLabels = testLabels(ind_test);


%clear numClasses;
%% Apply PCA to data

[total_data,~,~] = pca([trainData,testData]);
trainData = total_data(:,1:no_Train_samples);
testData = total_data(:,no_Train_samples+1 :no_Train_samples+no_Test_samples);

X = trainData' ;
y = trainLabels;
X_test =testData';
y_test = testLabels;


%% Setup the parameters you will use for this exercise

input_layer_size  = size(X_test,2);  
hidden_layer_size = ceil(input_layer_size/1.5);  
num_labels = numClasses;   

%% Initializing Parameters

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Train the NN

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 5000);


%  You should also try different values of lambda
% lambda = .01;

lambda = 15;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

             plot(cost) ;
fprintf('Program paused. Press enter to continue.\n');
pause;

%% prediction
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

  pred = predict(Theta1, Theta2, X_test);

 fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

 toc;






