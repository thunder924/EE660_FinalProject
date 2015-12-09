%% Logarithmic model
clear all
close all
clc

%% Increase the linearity of the nonlinear data by Logarithmic model
 % This file is used to test the Logarithmic model for the goal of achieve the
 % transformation from nonlinear data to linearity. 
 % It will run 20 times and the average MSE for test and training are:
 % Error_log
 % Error_tlog
 % Error_trec
load('RandomData.mat');
rs=0;
rs_t=0;
er=0;
er_t=0;
for I=1:20
    
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
trainX_n=standardizeCols(trainX);
testX_n=standardizeCols(testX);

trainX_log=log(trainX_n);
testX_log=log(testX_n);
model=linregFit(trainX_log,trainY);
res=linregPredict(model,testX_log);
res_t=linregPredict(model,trainX_log);
Res=round(res);
Res_t=round(res_t);

residual=Res-TestY;
residual_t=Res_t-trainY;
SStot=sum((TestY-mean(TestY)).^2);
SStot_t=sum((trainY-mean(trainY)).^2);
SSres=sum((Res-TestY).^2);
SSres_t=sum((Res_t-trainY).^2);
rs=rs+1-(SSres/SStot);
rs_t=rs_t+1-(SSres_t/SStot_t);
er=er+sum((residual).^2);
er_t=er+sum((residual_t).^2);

end
Er=er/20
Er_t=er_t/20
Rs=rs/20
Rs_t=rs_t/20
figure;
plot(TestY,residual,'+');title('Reciprocal model Residual');
    
