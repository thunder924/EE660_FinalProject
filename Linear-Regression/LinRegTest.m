clear all
close all
clc

load('RandomData.mat')

%% Linear regression with Original data
% Fit the linear regression model with original data and calculate the
% testing and training MSE.
% residual: residual of testing
% residual_t: residual of training
% SStot: total sum of squares of testing
% SStot_t: total sum of squares of training
% SSres: residual sum of squares of testing
% SSres_t: residual sum of squares of training
% Rs: R-squared for testing
% Rs_t: R-squared for training
% er: MSE of test
% er_t: MSE of training
% noise and noise2 are the noise for the TrainX and TestX
lambda=0.5;   
RandNum=randperm(35000,30000);
trainX=zeros(30000,59);
trainY=zeros(30000,1);
for i=1:30000
    trainX(i,:)=TrainX(RandNum(i),1:59);
    trainY(i)=TrainY(RandNum(i));
end
noise=randn(size(trainX))*10^-3;
noise2=randn(size(TestX))*10^-3;
trainX=trainX+noise;
testX=TestX+noise2;
model=linregFit(trainX,trainY);
res=round(linregPredict(model,testX));
res_t=round(linregPredict(model,trainX));
residual1=res-TestY;
residual_t1=res_t-trainY;
SStot1=sum((TestY-mean(TestY)).^2);
SStot_t1=sum((trainY-mean(trainY)).^2);
SSres1=sum((res-TestY).^2);
SSres_t1=sum((res_t-trainY).^2);
Rs1=1-(SSres1/SStot1)
Rs_t1=1-(SSres_t1/SStot_t1)
er1=sum((residual1).^2)
er_t1=sum((residual_t1).^2)
figure;
plot(TestY,residual1,'+');title('Original Data Residual');



%% With Normalization
trainX_n=standardizeCols(trainX);
testX_n=standardizeCols(testX);
model2=linregFit(trainX_n,trainY);
res2=round(linregPredict(model2,testX_n));
res2_t=round(linregPredict(model2,trainX_n));
residual2=res2-TestY;
residual_t2=res2_t-trainY;
SStot2=sum((TestY-mean(TestY)).^2);
SStot_t2=sum((trainY-mean(trainY)).^2);
SSres2=sum((res2-TestY).^2);
SSres_t2=sum((res2_t-trainY).^2);
Rs2=1-(SSres2/SStot2)
Rs_t2=1-(SSres_t2/SStot_t2)
er2=(sum((residual2).^2))/(length(res))
er_t2=(sum((residual_t2).^2))/(length(res_t))
figure;
plot(TestY,residual2,'+');title('Normalized Data Residual');

%% With Normalization, l2 regularization
model3=linregFit(trainX_n,trainY,'regType','L2','lambda',lambda);
res3=round(linregPredict(model3,testX_n));
res3_t=round(linregPredict(model3,trainX_n));
residual3=res3-TestY;
residual_t3=res3_t-trainY;
SStot3=sum((TestY-mean(TestY)).^2);
SStot_t3=sum((trainY-mean(trainY)).^2);
SSres3=sum((res3-TestY).^2);
SSres_t3=sum((res3_t-trainY).^2);
Rs3=1-(SSres3/SStot3)
Rs_t3=1-(SSres_t3/SStot_t3)
er3=sum((residual3).^2)
er_t3=sum((residual_t3).^2)
figure;
plot(TestY,residual3,'+');title('L2 regularized Data Residual');







