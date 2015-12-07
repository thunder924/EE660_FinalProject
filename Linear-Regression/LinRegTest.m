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
model=linregFit(trainX,TrainY,'regType','L2','lambda',lambda);
res=round(linregPredict(model,testX));
res_t=round(linregPredict(model,trainX));
er1=zeros(1,length(res));
for i=1:length(res);
    er1(i)=(res(i)-TestY(i))^2;
end
er1_t=zeros(1,length(res_t));
for i=1:length(res_t)
    er1_t(i)=(res_t(i)-TrainY(i))^2;
end
error1=(sum(er1))/length(res)
error1_t=(sum(er1_t))/length(res_t)


% Fit the linear regression model with normalized data and calculate the
% testing and training standard error.
% errror2: testing error for normalized data fitted linear regression
% errror2_t: training error for normalized data fitted linear regression
trainX_n=standardizeCols(TrainX);
testX_n=standardizeCols(TestX);


model2=linregFit(trainX_n,TrainY,'regType','L2','lambda',lambda);
res2=round(linregPredict(model2,testX_n));
res2_t=round(linregPredict(model2,trainX_n));

er2=zeros(1,length(res2));
for i=1:length(res2);
    er2(i)=(res2(i)-TestY(i))^2;
end
er2_t=zeros(1,length(res2_t));
for i=1:length(res2_t)
    er2_t(i)=(res2_t(i)-TrainY(i))^2;
end
error2=(sum(er2))/length(res2)
error2_t=(sum(er2_t))/length(res2_t)





