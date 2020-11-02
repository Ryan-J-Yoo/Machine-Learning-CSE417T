function [train_error] = find_train_error(w, X_train, y_train)

% find_test_error: compute the test error of a linear classifier w. The
%  hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
%  Inputs:
%		w: weight vector
%       X: data matrix (without an initial column of 1s)
%       y: data labels (plus or minus 1)
%     
%  Outputs:
%        test_error: binary error of w on the data set (X, y) error; 
%        this should be between 0 and 1. 

% w = zeros(d+1,1);
[N,d] = size(X_train);
A = ones(N,1);

for i = 1:N
    m = [A X_train]*w;
    h(i) = 1/(1 + exp(-m(i)));
    if h(i) >= 0.5
        h(i) = 1;
    else
        h(i) = -1;
    end
end

y_train(y_train == 0) = -1;


train_error = sum(sum(h(1,:)' ~= y_train(:,1)))/N;


end