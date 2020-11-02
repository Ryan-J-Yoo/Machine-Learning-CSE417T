function [t, w, e_in] = logistic_reg(X_train, y_train, w_init, max_its, eta)

% logistic_reg: learn a logistic regression model using gradient descent
%  Inputs:
%       X:       data matrix (without an initial column of 1s)
%       y:       data labels (plus or minus 1)
%       w_init:  initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta:     learning rate
%     
%  Outputs:
%        t:    the number of iterations gradient descent ran for
%        w:    learned weight vector
%        e_in: in-sample (cross-entropy) error 

t = 0;
w = w_init;
[N,d] = size(X_train);
%%%%%
%grad_sum = zeros([1 d+1]);
A = ones(N,1);

grad_mag = ones(N,d+1);
 
while max(grad_mag) > 1E-6
% for i = 1:max_its
    e_sum = 0;
    m = [A X_train]*w;
    a = [A X_train];
    
    for n = 1:N
    grad_sum(n,1:d+1) = -y_train(n)*a(n,:)/((1 + exp(y_train(n)*m(n)))*N);
    e_sum = e_sum + log(1 + exp(-y_train(n)*m(n))); 
    end

    e_in = (e_sum)/N;
    grad_e_in = sum(grad_sum);
    grad_mag = abs(grad_e_in);

%     if grad_mag < 0.000001
%          break;
%      end
     
    t = t + 1;   
    w = w - eta*(grad_e_in)';

end
   
% end



end

