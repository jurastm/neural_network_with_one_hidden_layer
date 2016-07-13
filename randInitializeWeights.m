function W = randInitializeWeights(L_in, L_out)

%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections

W = zeros(L_out, 1 + L_in);
% Note: The first row of W corresponds to the parameters for the bias units

epsilon = sqrt(6)/sqrt(L_in + L_out);
W = rand(L_out, 1 + L_in)*2*epsilon - epsilon;

end
