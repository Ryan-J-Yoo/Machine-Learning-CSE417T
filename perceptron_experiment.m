function [num_iters, bounds_minus_ni] = perceptron_experiment(N, d, num_samples)
% perceptron_experiment: Code for running the perceptron experiment in HW1
% Inputs:  N is the number of training examples
%          d is the dimensionality of each example (before adding the 1)
%          num_samples is the number of times to repeat the experiment
% Outputs: num_iters is the # of iterations PLA takes for each sample
%          bound_minus_ni is the difference between the theoretical bound
%                         and the actual number of iterations
%          (both the outputs should be num_samples long)

    
  for i=1:num_samples      
    w_real = [0;rand(d,1)];
    X = 2*rand(N,d+1) - 1;
    X(:,1) = ones;
    y_real = sign(X * w_real);
    
    data_in = [X y_real];
    
    [w, it, y_predict] = perceptron_learn(data_in);
    
    
    num_iters(i) = it;
    
    a = norm(w_real);
    R = max(norm(X));
    size(y_predict)
    size(w_real)
    size(X)
    rho = min(y_predict'.*(w_real'*X'))
  
    bounds_minus_ni(i) = a^2 * R^2 / rho^2 - it;
  end  
  
end