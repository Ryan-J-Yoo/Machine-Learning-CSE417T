function [test_error] = find_test_error(w, X_test, y_test)

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
[N,d] = size(X_test);
A = ones(N,1);

for i = 1:N
    m = [A X_test]*w;
    h(i) = 1/(1 + exp(-m(i)));
    if h(i) >= 0.5
        h(i) = 1;
    else
        h(i) = -1;
    end
end

y_test(y_test == 0) = -1;


test_error = sum(sum(h(1,:)' ~= y_test(:,1)))/N;


end