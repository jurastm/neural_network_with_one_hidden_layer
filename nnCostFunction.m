function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a one hidden layer
%neural network which performs classification


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),  num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));




%Feeding forward propagation
a1 = [ones(m,1) X];               % add "ones" to input layer
z2 = a1*Theta1';                  % compute linear z-function for hidden layer
a2 = [ones(m,1) sigmoid(z2)];     % add "ones" hidden layer, and compute sigmoid function for hidden layer
z3 = a2*Theta2';                  %  compute z-function for output layer
a3 = sigmoid(z3);                 %  compute output layer hypotesis

yd = eye(num_labels);            % these variables convert output from {0, 1} value
yVector = yd(y, :);              % to output vectors e.g. [0 0...  0 1 0], where 1 indicates predicted class


ThetaT1 = Theta1;     % setup ThetaT matrixes with 
ThetaT2 = Theta2;     % zeroed first element
ThetaT1(:, 1) = 0;    % it is needed for correct 
ThetaT2(:, 1) = 0;    % gradients and cost function computation 

reg = (lambda/(2*m))*((sum(sum(ThetaT1 .^2))) + sum(sum(ThetaT2 .^2)));  %regularization part of cost function
cost = (-yVector .* log(a3)) - ((1 - yVector) .* log(1 - a3));           % cost function
J = (sum(sum(cost))/m) + reg;                                            % cost function

%Vectorized backpropagation
delta3 = a3 - yVector;                                              % compute deltas of output layer
delta2 = (delta3*Theta2) .* sigmoidGradient([ones(m, 1), z2]);      % compute deltas of hidden layer 

Delta3 = delta3' * a2;                % accumulate and storing 
Delta2 = (delta2(:, 2:end))' *a1;     % delta parameters

Theta1_grad = (1/m)*Delta2 + (lambda/m)*ThetaT1;     % compute gradients
Theta2_grad = (1/m)*Delta3 + (lambda/m)*ThetaT2;     % compute gradients

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
