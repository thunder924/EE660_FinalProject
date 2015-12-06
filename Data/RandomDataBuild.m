clear all
close all
clc

% This function is used to select the Training data and test data randomly,
% if you want to save the data to a new file, please uncomment the "save"
% part.
% TrainX: training input data
% TrainY: training output data
% TestX: testing input data
% TestY: testing output data

load('OriginalData.mat');
RandNum=randperm(39644,35644);
TrainX=zeros(35644,59);
TrainY=zeros(35644,1);
for i=1:35644
    TrainX(i,:)=DATA(RandNum(i),1:59);
    TrainY(i)=DATA(RandNum(i),60);
end
RandNum2=[];
for j=1:39644
    if any(j==RandNum)==0
        RandNum2=[RandNum2 j];
    end
end
TestX=zeros(4000,59);
TestY=zeros(4000,1);
for k=1:length(RandNum2)
    TestX(k,:)=DATA(RandNum2(k),1:59);
    TestY(k)=DATA(RandNum2(k),60);
end

% If you want to generate new data, just uncomment below:
% save RandomData TrainX TrainY TestX TestY

