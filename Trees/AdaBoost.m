function [train_err, test_err] = AdaBoost(X_tr, y_tr, X_te, y_te, numTrees)
% AdaBoost: Implement AdaBoost with decision stumps as the weak learners. 
%           (hint: read the "Name-Value Pair Arguments" part of the "fitctree" documentation)
%   Inputs:
%           X_tr: Training data
%           y_tr: Training labels
%           X_te: Testing data
%           y_te: Testing labels
%           numTrees: The number of trees to use
%  Outputs: 
%           train_err: Classification error of the learned ensemble on the training data
%           test_err: Classification error of the learned ensemble on test data
% 
% You may use "fitctree" but not any inbuilt boosting function

    % Initialization
    n_tr = size(X_tr,1);
    n_te = size(X_te,1);
    w = (1/n_tr) .* ones(1, n_tr);    % all weights sum to 1  
    gt_tr = zeros(n_tr,1);            % traning output hypothesis
    gt_te = zeros(n_te,1);            % test output hypothesis
    train_err = zeros(numTrees,1);    % training error list 
    test_err = zeros(numTrees,1);     % test error list 
    
    % Modify y for Classification
    pick = [1 -1];
    label = sort(unique(y_tr));
    id = label(1)*label(2);
    y_tr(y_tr==label(1)) = -1;
    y_tr(y_tr==label(2)) =  1;
    y_te(y_te==label(1)) = -1;
    y_te(y_te==label(2)) =  1;
    
    for t = 1:numTrees
        
        % train a weak learner ht
        ht = fitctree(X_tr,y_tr, 'Weights', w',...
                     'SplitCriterion','deviance','MaxNumSplits',1);
        y_pred_tr = predict(ht, X_tr);
        y_pred_te = predict(ht, X_te);
        % computing thw weighted training error of ht
        et = w * (y_pred_tr ~= y_tr);
        % compute the "importance" of ht
        at = 0.5*log((1-et)/et);
        % update weights
        Zt = 2 * sqrt(et * (1 - et));     % normalization parameters
        w = (w./Zt) .* (exp(-at) .* (y_pred_tr == y_tr) + ...
                        exp( at) .* (y_pred_tr ~= y_tr))';
        
        % output classification result by aggregating hypothesis
        gt_tr = gt_tr + at * y_pred_tr;
        gt_te = gt_te + at * y_pred_te;
        
        % break tie by random classification
        gt_tr(gt_tr(:) == 0) = pick(randperm(2, 1));
        gt_te(gt_te(:) == 0) = pick(randperm(2, 1));
        
        % store traning error and test error
        train_err(t) = mean(sign(gt_tr) ~= y_tr);
        test_err(t) = mean(sign(gt_te) ~= y_te);
        
    end    
    
    figure()
    plot(1:numTrees, train_err,'linewidth', 2)
    hold on
    plot(1:numTrees, test_err,'linewidth', 2)
    xlabel('Number of Trees');ylabel('Error');
    legend('Training Error','Test Error');
    if id == 3
        title('AdaBoost Problem 1 : Digit One vs Digit Three');
    else
        title('AdaBoost Problem 2 : Digit Three vs Digit Five');
    end
    grid on;
    
end