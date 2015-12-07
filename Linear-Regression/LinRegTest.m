clear all
close all
clc

load('RandomData.mat')

% Fit the linear regression model with original data and calculate the
% testing and training standard error.
% errror1: testing error for original data fitted linear regression
% errror1_t: training error for original data fitted linear regression
% noise and noise2 are the noise for the TrainX and TestX
lambda=10;
noise=randn(size(TrainX))*10^-3;
noise2=randn(size(TestX))*10^-3;
trainX=TrainX+noise;
testX=TestX+noise2;
model=linregFit(trainX,TrainY);
res=round(linregPredict(model,testX));
res_t=round(linregPredict(model,trainX));
error1=sum((res-TestY).^2)
error1_t=sum((res_t-TrainY).^2)


% Fit the linear regression model with normalized data and calculate the
% testing and training standard error.
% errror2: testing error for normalized data fitted linear regression
% errror2_t: training error for normalized data fitted linear regression
trainX_n=standardizeCols(trainX);
testX_n=standardizeCols(testX);
model2=linregFit(trainX_n,TrainY);
res2=round(linregPredict(model2,testX_n));
res2_t=round(linregPredict(model2,trainX_n));
error2=sum((res2-TestY).^2)
error2_t=sum((res2_t-TrainY).^2)





