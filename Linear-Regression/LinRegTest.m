clear all
close all
clc

load('RandomData.mat')

%% Linear regression with Original data
% Fit the linear regression model with original data and calculate the
% testing and training standard error.
% errror: testing error for original data fitted linear regression
% errror_t: training error for original data fitted linear regression
% noise and noise2 are the noise for the TrainX and TestX
lambda=1;
noise=randn(size(TrainX))*10^-3;
noise2=randn(size(TestX))*10^-3;
trainX=TrainX+noise;
testX=TestX+noise2;
model=linregFit(trainX,TrainY);
res=round(linregPredict(model,testX));
res_t=round(linregPredict(model,trainX));
error1=(sum((res-TestY).^2))/length(res)
error1_t=(sum((res_t-TrainY).^2))/length(res_t)


%% With Normalization
trainX_n=standardizeCols(trainX);
testX_n=standardizeCols(testX);
model2=linregFit(trainX_n,TrainY);
res2=round(linregPredict(model2,testX_n));
res2_t=round(linregPredict(model2,trainX_n));
error2=(sum((res2-TestY).^2))/length(res2)
error2_t=(sum((res2_t-TrainY).^2))/length(res2_t)

%% With Normalization, l2 regularization
model3=linregFit(trainX_n,TrainY,'regType','L2','lambda',lambda);
res3=round(linregPredict(model3,testX_n));
res3_t=round(linregPredict(model3,trainX_n));
error3=(sum((res3-TestY).^2))/length(res3)
error3_t=(sum((res3_t-TrainY).^2))/length(res3_t)

%% With Normalization, l2 regularization, student model
model4=linregFit(trainX_n,TrainY,'regType','L2','lambda',lambda,'likelihood','student');
res4=round(linregPredict(model4,testX_n));
res4_t=round(linregPredict(model4,trainX_n));
error4=(sum((res4-TestY).^2))/length(res4)
error4_t=(sum((res4_t-TrainY).^2))/length(res4_t)







