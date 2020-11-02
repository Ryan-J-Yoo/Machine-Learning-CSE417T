% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)
clear;

zip_train = table2array(readtable('zip_train.csv'));
zip_test = table2array(readtable('zip_test.csv'));

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip_train(find(zip_train(:,1) == 1 | zip_train(:,1) == 3),:);
X_tr = subsample(:,2:257);
y_tr = subsample(:,1);
subsample = zip_test(find(zip_test(:,1) == 1 | zip_test(:,1) == 3),:);
X_te = subsample(:,2:257);
y_te = subsample(:,1);

ct = fitctree(X_tr, y_tr, 'CrossVal', 'on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
t = fitctree(X_tr, y_tr);
fprintf('The test error of decision trees is %.4f\n', sum(predict(t,X_te) ~= y_te)/length(y_te));
[oobErr_bagTrees, testErr_bagTrees] = BaggedTrees(X_tr, y_tr, X_te, y_te, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', oobErr_bagTrees(end));
fprintf('The test error of 200 bagged decision trees is %.4f\n', testErr_bagTrees);
[oobErr_randForest, testErr_randForest] = RandomForest(X_tr, y_tr, X_te, y_te, 200, floor(size(X_tr, 2)/3));
fprintf('The OOB error of the random forest is %.4f\n', oobErr_randForest(end));
fprintf('The test error of the random forest is %.4f\n', testErr_randForest);

fprintf('\n');

fprintf('Now working on the three-vs-five problem...\n\n');
subsample = zip_train(find(zip_train(:,1) == 3 | zip_train(:,1) == 5),:);
X_tr = subsample(:,2:257);
y_tr = subsample(:,1);
subsample = zip_test(find(zip_test(:,1) == 3 | zip_test(:,1) == 5),:);
X_te = subsample(:,2:257);
y_te = subsample(:,1);

ct = fitctree(X_tr, y_tr, 'CrossVal', 'on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
t = fitctree(X_tr, y_tr);
fprintf('The test error of decision trees is %.4f\n', sum(predict(t,X_te) ~= y_te)/length(y_te));
[oobErr_bagTrees, testErr_bagTrees] = BaggedTrees(X_tr, y_tr, X_te, y_te, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', oobErr_bagTrees(end));
fprintf('The test error of 200 bagged decision trees is %.4f\n', testErr_bagTrees);
[oobErr_randForest, testErr_randForest] = RandomForest(X_tr, y_tr, X_te, y_te, 200, floor(size(X_tr, 2)/3));
fprintf('The OOB error of the random forest is %.4f\n', oobErr_randForest(end));
fprintf('The test error of the random forest is %.4f\n', testErr_randForest);