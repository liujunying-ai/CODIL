function [ AvgF1,AvgP,AvgR,Acc ] = MultiClassMetric( y_predict,y_ground )
% MultiClassMetric computes the Average F1, Average Precision, Average Recall and Accuracy for multi-class data
%
%     [ AvgF1,AvgP,AvgR,Acc ] = MultiClassMetric( y_predict,y_ground )
%
% Description
%
%       MultiClassMetric takes,
%           y_predict	- The predicted class vector
%           y_ground	- The ground-truth class vector
%      
%       and returns,
%           AvgF1       - Average F1
%           AvgP        - Average Precision
%           AvgR        - Average Recall
%           Acc         - Accuracy
%  
    class_set = unique(y_ground);
    MetricP = zeros(length(class_set),1);
    MetricR = zeros(length(class_set),1);
    MetricF1 = zeros(length(class_set),1);
    for ii=1:length(class_set)
        tmp_yp = (y_predict==class_set(ii));
        tmp_yg = (y_ground==class_set(ii));
        numTP =  sum(tmp_yp&tmp_yg);
        %numTN =  sum((~tmp_yp)&(~tmp_yg));
        numFP =  sum((tmp_yp)&(~tmp_yg));
        numFN =  sum((~tmp_yp)&(tmp_yg));
        if numTP+numFP==0
            MetricP(ii) = 0;
        else
            MetricP(ii) = numTP/(numTP+numFP);
        end
        if numTP+numFN==0
            MetricR(ii) = 0;
        else
            MetricR(ii) = numTP/(numTP+numFN);
        end
        if MetricP(ii)+MetricR(ii)==0
            MetricF1(ii) = 0;
        else
            MetricF1(ii) = 2*MetricP(ii)*MetricR(ii)/(MetricP(ii)+MetricR(ii));
        end
    end
    AvgP = mean(MetricP);
    AvgR = mean(MetricR);
    AvgF1 = mean(MetricF1);
    Acc = sum(y_predict==y_ground)/length(y_ground);
end

