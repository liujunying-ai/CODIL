function [ Eval,y_predict ] = CODIL(X_train, y_train, X_test, y_test, NumK)
%CODIL implements the CODIL approach as described in [1]
%Type 'help CODIL' under Matlab prompt for more detailed information about CODIL
%
%	Syntax
%
%       [ Eval,y_predict ] = CODIL(X_train, y_train, X_test, y_test, NumK)
%
%	Description
%
%   CODIL takes,
%       X_train     - An mxd matrix, the ith instance of training instance is stored in X_train(i,:)
%       y_train     - An mx1 vector, the ith class label of training instance is stored in y_train(i)
%       X_test      - An pxd matrix, the ith instance of testing instance is stored in X_test(i,:)
%       y_test      - An px1 vector, the ith class label of testing instance is stored in y_test(i,:)
%       NumK        - Number of nearest neighbors considered (default 12)
%   and returns,
%       Eval	    - A struct where 
%						Eval.ACC correpsonds to the Accuracy on testing data as described in [1]
%						Eval.AvgF1 correpsonds to the Average-F1 on testing data as described in [1]
%       y_predict	- An px1 array, the predicted class vector for test instance matrix X_test
%
%  [1] J.-Y. Liu, B.-B. Jia. Combining One-vs-One Decomposition and Instance-based Learning for Multi-Class Classification, In: IEEE Access,vol.8, pp.197499 - 197507, 2020.
%
%See also MultiClassMetric.

    % Default parameter setting
    if nargin<5
        NumK = 12;
    end
    
    % Obtain parameters of data sets
    num_training = size(X_train,1);%number of training examples
    num_testing = size(X_test,1);%number of testing examples
    C_label = unique(y_train);%unique class labels
    num_label = length(C_label);%number of class labels
    
    % One-vs-One (OvO) decomposition
    num_ovo = num_label.*(num_label-1)/2;
    y_train_ovo = zeros(num_training,num_ovo);
    y_code_ovo = zeros(2,num_ovo);
    cnt_ovo = 1;
    for a1=1:num_label-1
        label_p = C_label(a1);
        for a2=a1+1:num_label
            label_n = C_label(a2);
            tmp_y = zeros(num_training,1);
            tmp_y(y_train==label_p) = +1;
            tmp_y(y_train==label_n) = -1;
            y_train_ovo(:,cnt_ovo) = tmp_y;
            %save OvO code table
            y_code_ovo(1,cnt_ovo) = label_p;
            y_code_ovo(2,cnt_ovo) = label_n;
            cnt_ovo = cnt_ovo + 1;
        end
    end
    
    % main
    [idx_test, dist_test] = knnsearch(X_train,X_test,'k',NumK);
    F_test = zeros(num_testing,num_ovo);
    for itest=1:num_testing
        % Determine the linear combination coefficient by solving the optimization problem in Eq.(3)
        dist0_idx = (dist_test(itest,:)==0);
        if sum(dist0_idx)>0%if there are training instances which are identical to the itest-th testing instance
            si = zeros(NumK,1);
            si(dist0_idx) = 1/sum(dist0_idx);
        else
            XK = transpose(X_train(idx_test(itest,:),:));
            Xi = repmat(transpose(X_test(itest,:)),1,NumK);
            Ci = (Xi-XK)'*(Xi-XK);
            opts = optimoptions('quadprog','Display','off');
            H = 2*(Ci'+Ci)/2;
            f = zeros(NumK,1);
            Aeq = ones(1,NumK);
            beq = 1;
            lb = zeros(NumK,1);
            si = quadprog(H,f,[],[],Aeq,beq,lb,[],[],opts);
        end
        F_train_K = y_train_ovo(idx_test(itest,:),:);
        F_test(itest,:) = transpose(si)*F_train_K;% Obtain the real-valued label vector for unseen instance
    end
    
    %One-vs-One (OvO) decoding
    y_predict = zeros(size(y_test));
    for ii=1:num_testing
        tmp_y_test_ii = zeros(1,num_ovo);
        for jj=1:num_ovo
            if F_test(ii,jj)>=0%sign operation in Eq.(4)
                tmp_y_test_ii(jj) = y_code_ovo(1,jj);
            else
                tmp_y_test_ii(jj) = y_code_ovo(2,jj);
            end
        end
        tmp_cnt = zeros(num_label,1);
        for jj=1:num_label
            tmp_cnt(jj) = sum(tmp_y_test_ii == C_label(jj));
        end
        [p_val,p_pos] = max(tmp_cnt);%majority voting
        y_predict(ii) = C_label(p_pos);
    end
    
    % Evaluation metric
    [Eval.AvgF1,~,~,Eval.ACC] = MultiClassMetric(y_predict,y_test);    
end

