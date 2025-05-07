%This is an examplar file on how the CODIL program could be used (The main function is "CODIL.m")
%
%Type 'help CODIL' under Matlab prompt for more detailed information

clc;clear;close;
% Load the file containing the necessary inputs for calling the CODIL function
load('sample data.mat'); 

% The number of nearest neighbors considered by CODIL
NumK=12;

% Calling the main function CODIL
[ Eval,y_predict ] = CODIL( X_train, y_train, X_test, y_test, NumK );
disp(['Accuracy=',num2str(Eval.ACC,'%4.3f'),', Average-F1=',num2str(Eval.AvgF1,'%4.3f')]);