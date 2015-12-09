clear all
close all
clc

%% Random Forest

load('RandomData.mat');
  
RandNum=randperm(35000,30000);
trainX=zeros(30000,59);
trainY=zeros(30000,1);
for i=1:30000
    trainX(i,:)=TrainX(RandNum(i),1:59);
    trainY(i)=TrainY(RandNum(i));
end

model1=regRF_train(trainX,trainY,50);
res=regRF_predict(TestX,model1);
res_t=regRF_predict(trainX,model1);

residual=res-TestY;
residual_t=res_t-trainY;
SStot=sum((TestY-mean(TestY)).^2);
SStot_t=sum((trainY-mean(trainY)).^2);
SSres=sum((res-TestY).^2);
SSres_t=sum((res_t-trainY).^2);
rs=1-(SSres/SStot)
rs_t=1-(SSres_t/SStot_t)
er=sum((residual).^2)
er_t=sum((residual_t).^2)

