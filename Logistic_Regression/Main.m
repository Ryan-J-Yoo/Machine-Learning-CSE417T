clear;
clc;

data_train = table2array(readtable('cleveland_train.csv'));
d = 13;
X_train = data_train(:,1:d);
y_train = data_train(:,14);
w_init = zeros(d+1,1);
max_its = 1e6;
eta = 7.7;
y_train(y_train == 0) = -1;

[X_train, mu, sigma] = zscore(X_train, 0, 1);



tic
[t, w, e_in] = logistic_reg(X_train, y_train, w_init, max_its, eta)
toc

data_test = table2array(readtable('cleveland_test.csv'));
X_test = data_test(:,1:d);
y_test = data_test(:,14);

X_test = (X_test - mu)./sigma;

[test_error] = find_test_error(w, X_test, y_test)

[train_error] = find_train_error(w, X_train, y_train)

