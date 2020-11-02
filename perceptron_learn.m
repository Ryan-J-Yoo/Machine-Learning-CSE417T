function [w, iterations,y_predict] = perceptron_learn(data_in)
% perceptron_learn: Run PLA on the input data
% Inputs:  data_in is a matrix with each row representing an (x,y) pair;
%                 the x vector is augmented with a leading 1,
%                 the label, y, is in the last column
% Outputs: w is the learned weight vector; 
%            it should linearly separate the data if it is linearly separable
%          iterations is the number of iterations the algorithm ran for
 
w = zeros(11,1);
 
X = data_in(:,1:11);
y_real = data_in(:,12);
 
iterations = 0;
y_predict = ones(100,1);
 
flag = true;

while flag
    
    flag = false;
    for i = 1:100
        if (y_predict(i) < y_real(i))||(y_predict(i) > y_real(i))
            %disp(y_predict(i))
            w = w + y_real(i) * X(i,:)';
            iterations = iterations + 1;
            y_predict = sign(X*w);
            flag = true;
        end
    end
 
end
