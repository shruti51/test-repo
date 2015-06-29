function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
                               
  
  
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); 
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% decoding the output labels :
  
     output_labels = zeros(m,num_labels) ;
     
     for i = 1 : m
         
         output_labels(i,y(i))=1;
         
     end
     
%      output_labels
     
 % Decoding done ! Now use output_labels in place of Y.    

      a1 = [ones(m,1) X];
      
      z2 = a1 * Theta1' ;
      
      a2 = [ones(size(z2,1),1) sigmoid(z2)];
      
      z3 = a2 * Theta2' ;
      
      a3 = sigmoid(z3);
       
%       size(a3)
%       size(output_labels)
%       
      
   % delta_3 = output_labels - a3;
     
      J_temp =0;
      for i = 1 : m
          q = -output_labels(i,:) .* log(a3(i,:));
          J_temp = J_temp  +sum(q - (1 - output_labels(i,:)).*log(1-a3(i,:)));
      end
      
      J=(1/m)*J_temp;
      
      
      % regularized  parameter
      
      J_reg = (sum(sum(Theta1 .^ 2)) +sum(sum(Theta2 .^ 2)));
      J = J + J_reg*(lambda/(2*m));
      




% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

 %num_hidden_layers=1;

 %delta_accumulator_1 = zeros(hidden_layer_size,input_layer_size+1);
 %delta_accumulator_2 = zeros(num_labels,hidden_layer_size+1);
 
 for i = 1 : m
     a1 = [1 ; X(i,:)']; % dim -> (n+1) x 1
     z2 = Theta1 * a1 ;
     a2 = sigmoid(z2) ; % dim -> hidden_layer_size x 1
     a2 = [1 ; a2];
     z3 = Theta2 * a2 ;
     a3 = sigmoid(z3); % dim -> num_labels x 1
     
    % yy = ([1:num_labels]==y(i))';
     delta_3 = a3 - output_labels(i,:)'  ;% dim -> num_labels x 1
    % delta_3 = a3 - yy;
     delta_2 = Theta2' * delta_3 ; 
     delta_2 = delta_2 .* sigmoidGradient([1 ; z2]) ;  % dim -> hid_layer_size+1 x 1
      % skiping the first element of delta_2 as it doesnot need to
      % backpropgate anything (see figure)
     delta_2 = delta_2(2:end); % dim -> hid_layer_size x 1
     
     Theta1_grad = Theta1_grad + delta_2 * a1'; % dim -> hidden_layer_size x input_size+1
     Theta2_grad = Theta2_grad + delta_3 * a2'; % dim -> num_labels x hidden_layer_size+1
     
       
     
 end
 
 %Theta1_grad = delta_accumulator_1 ./ m;
 %Theta2_grad = delta_accumulator_2 ./ m;

  %Theta1_grad = Theta1_grad;
  %Theta2_grad = Theta_grad2;






%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%Theta1_grad = Theta1_grad + (lambda/m) .*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
%Theta2_grad = Theta2_grad + (lambda/m) .*[zeros(size(Theta2,1),1) Theta2(:,2:end)];






Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
