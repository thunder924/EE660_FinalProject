% This function is used to test the simple linear regression with the original data and normalized data added the Gaussian noise

clear all
close all
clc

load('RandomData.mat')


noise=randn(size(TrainX))*10^-3;
noise2=randn(size(TestX))*10^-3;
trainX=TrainX+noise;
testX=TestX+noise2;
model=linregFit(trainX,TrainY);
res=round(linregPredict(model,testX));
er1=(length(find(res~=TestY)))/(length(res))

[r,c]=size(trainX);
y=zeros(r,c);
for i=1:c
    v=trainX(:,i);
    m=mean(v);
    s=std(v);
    y(:,i)=(v-m)/s;
end

[r2,c2]=size(testX);
y2=zeros(r2,c2);
for k=1:c2
    v2=testX(:,k);
    m2=mean(v2);
    s2=std(v2);
    y2(:,k)=(v2-m2)/s2;
end

model2=linregFit(y,TrainY);
res2=round(linregPredict(model2,y2));
er2=(length(find(res2~=TestY)))/(length(res2))



