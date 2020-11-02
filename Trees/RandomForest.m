function [oob_err, test_err] = RandomForest(X_tr, y_tr, X_te, y_te, numBags, m)
% RandomForest: Learns an ensemble of numBags CART decision trees using a random subset of
%               the features at each split on the input dataset and also plots the 
%               out-of-bag error as a function of the number of bags
%       Inputs:
%               X_tr: Training data
%               y_tr: Training labels
%               X_te: Testing data
%               y_te: Testing labels
%               numBags: Number of trees to learn in the ensemble
% 				m: Number of randomly selected features to consider at each split
% 				   (hint: read the "Name-Value Pair Arguments" part of the fitctree documentation)
%      Outputs: 
%	            oob_err: Out-of-bag classification error of the final learned ensemble
%               test_err: Classification error of the final learned ensemble on test data
%
% You may use "fitctree" but not "TreeBagger" or any other inbuilt bagging function

    % Modify y for Classification
    pick = [1 -1];
    label = sort(unique(y_tr));
    id = label(1) * label(2);
    y_tr(y_tr==label(1)) = -1;
    y_tr(y_tr==label(2)) =  1;
    y_te(y_te==label(1)) = -1;
    y_te(y_te==label(2)) =  1;
    
    % Initialization
    vote_tr = zeros(size(X_tr,1),1);
    vote_te = zeros(size(X_te,1),1);
    oob_err = zeros(numBags,1);
    valid_vote_tr = vote_tr;
    
    
    for b = 1:numBags
        
        % Bootstapping
        dataIndex = 1:size(X_tr,1);
        bagIndex = datasample(dataIndex, size(X_tr,1),'Replace', true);
        
        % Learn Decision Tree (CART Algorithm)
        yBag = y_tr(bagIndex, :);
        xBag = X_tr(bagIndex, :);
        tree = fitctree(xBag,yBag, 'NumVariablesToSample',m);
        
        % Finding X not used to train
        outIndex = setdiff(dataIndex, bagIndex);
        X_out = X_tr(outIndex,:);
        
        % Aggregated Hypothesis (Vote)
        vote_tr(outIndex) = vote_tr(outIndex) + predict(tree,X_out);  
        
        % Make Random Choice then Output Y
        valid_vote_tr(outIndex) = vote_tr(outIndex);
        valid_vote_tr(vote_tr == 0) = pick(randperm(2, 1));
        valid_vote_tr = sign(valid_vote_tr);
        
        % Out-of-bag Error
        oob_err(b) = mean(valid_vote_tr ~= y_tr);
        
        % Test
        % Aggregated Hypothesis (Vote)
        vote_te = vote_te + predict(tree,X_te);
        
        if b == numBags
            % Break Tie and Output Y
            vote_te(vote_te == 0) = pick(randperm(2, 1));
            vote_te = sign(vote_te);
            % Classification Error
            test_err = mean(vote_te ~= y_te);
        end
    end  
    
    figure();
    plot(1:numBags,oob_err,'linewidth', 2);
    xlabel('Number of Bags');ylabel('Out-of-bag Error');
    if id == 3
        title('Random Forest Problem 1 : Digit One vs Digit Three');
    else
        title('Random Forest Problem 2 : Digit Three vs Digit Five');
    end
    grid on; 
    
end
