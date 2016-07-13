# neural_network_with_one_hidden_layer
Here is implementation of neural network with one hidden layer. It has vectorized implementation of backpropagation.

Consequense of implementation

1. Load your labeled data.
2. Setup desirable hidden_layer_size
3. Define input layer size. If picture has size 10X10 pixels, then your input layer size is 10*10 = 100 pixels. If your picture is RGB colored, then input layer size will be 10*10*3 = 300.
4. define numeber of labels with vaiable num_labels
5. define regularization parameter lambda
6. define initial_Theta1 and initial_Theta2. For this purpose use randInitializeWeights.m, it need for symmetry breaking
7. Unroll initial Thetas to a single vector initial_nn_params = initial_Theta1(:) ; initial_Theta2(:)];
8. Before implementing backpropagation it is useful to make gradient checking, for this use checkNNGradients.m. You can check gradients with and without regularization
9. Teach Neural network. For this purpose I use very advanced L-BFGS algorithm that stored into fmincg.m function. First you have to assign options = optimset('MaxIter', 50) - whereas you set maximal amount of iterations. Then assign: 
          costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

10. Minimize cost function by fmincg.m (L-BFGS) [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
11. Reshape Thetas from single vector to matrixes: 
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

12. Finally you can make your predictions for new examples from cross validation set and play with lambda to gain best results on cross validation set.
 pred = predict(Theta1, Theta2, new_example);


