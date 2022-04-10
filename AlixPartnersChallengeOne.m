function [AllX,AllY,T,OptLatents,Gp0_AUC_plsDA,Gp1_AUC_plsDA,Fold_Gp0_AUC,Fold_Gp1_AUC]=AlixPartnersChallengeOne(fn)
%This code is written to try and solve the Alix Partners Challenge.
%Inputs
%fn - string input for the .csv file downloaded for the original challenge
%
%Outputs
% AllX - The PLS dimensionally regressed input variables for all 20,000 data points (should be down to 3 dimensions)
% AllY - The PLS predictors for for all 20,000 data points
% T - the original csv file now in a table format (used in the classification learner app)
% OptLatents - The optimum number of latents determined by the PLS regression
% Gp0_AUC_plsDA - The area under the curve for ROC plots after PLS-DA for group 0
% Gp1_AUC_plsDA - The area under the curve for ROC plots after PLS-DA for group 1
% Fold_Gp0_AUC - The area under the curve for ROC plots after PLS-SVM for group 0, values are for each fold
% Fold_Gp1_AUC - The area under the curve for ROC plots after PLS-SVM for group 1, values are for each fold
% 
% Additionally a csv file titled "OutputTable.csv" is created with 19,750
% rows, the first column being the row id, the second being the predicted
% probabilities.
%
% Created by Marcus Leiwe - Oct 2021
%% Load inputs and break down
T=readtable(fn);
AllVarOnly=table2array(T(:,4:303));
ArrayT=table2array(T);
TotalN=size(AllVarOnly,1);
%Just group 0
Gp0_idx=ArrayT(:,3)==0;
%Just group 1
Gp1_idx=ArrayT(:,3)==1;
%All Group 0 and 1
AllGpIdx=logical(Gp0_idx+Gp1_idx);
AllTest=ArrayT(AllGpIdx,:);
szAllTestSet=sum(Gp0_idx)+sum(Gp1_idx);
%% Show that PCA won't work
[~,~,~,~,explained,~]=pca(AllVarOnly);
figure('Name','Unbiassed Dimension Reduction with PCA')
plot(1:300,cumsum(explained),'-bo');
ylabel('Cumulative Variance Explained (%)')
xlabel('Number of Components')
title('Variance Explained By PCA')
%% Show that in theory PLS regression can work
[Xl,Yl,Xs,Ys,beta,pctVar,PLSmse]=plsregress(AllTest(:,4:303),AllTest(:,3),szAllTestSet-1);
figure('Name','Initial Test Settings')
subplot(1,2,2)
plot(1:szAllTestSet,PLSmse(2,:),'-bo');
title('MSE')
xlabel('Number of Latents')
ylabel('MSE')
xlim([0 szAllTestSet])
subplot(1,2,1);
plot(1:szAllTestSet-1,cumsum(100*pctVar(2,:)),'-bo');
xlim([0 szAllTestSet-1])
ylim([0 100])
title('Variance Explained')
xlabel('Number of Latents')
ylabel('Percent of Variance Explained in Y')
%% Assess via cross validation
nFolds=[2 3 4 5:5:round(szAllTestSet/5)];
maxLatents=50;
[~,OptLatents]=mnl_AssessPLSbyCV(AllTest,nFolds,maxLatents); %Decided not to use the optimum folds calculation
%% Now create a model for predicting how good the final DA model will be
nFolds=15; %Decided to use 15 folds for optimisation
[Gp0_AUC_plsDA,Gp1_AUC_plsDA]=mnl_AssessPLSbyDA(AllTest,nFolds,OptLatents);
%% Now create a model of the full training set
%PLS regress based on the whole train set
[Xl_final,Yl_final,Xs_final,Ys_final,beta_final,pctVar_final,PLSmse_final]=plsregress(AllTest(:,4:303),AllTest(:,3),OptLatents,'cv',nFolds);
%Transform all input variables
AllX=AllVarOnly*Xl_final;
%Predict the output
x=[ones(TotalN,1) AllVarOnly];
AllY=x*beta_final;
%Plot the data so we can visualise it
figure('Name','3D regression of the data')
%Plot all the points
scatter3(AllX(:,1),AllX(:,2),AllX(:,3),2,[0.5 0.5 0.5],'filled')
hold on
%Plot those known as group 0
scatter3(AllX(Gp0_idx,1),AllX(Gp0_idx,2),AllX(Gp0_idx,3),10,'k','filled')
%Plot those known as group 1
scatter3(AllX(Gp1_idx,1),AllX(Gp1_idx,2),AllX(Gp1_idx,3),10,'r','filled')
%% Okay we've evaluated the which machine learning algorithm works best and settled on an SVM
%Final assessment of the SVM classifier by cross validation
All_Data=[AllX(1:szAllTestSet,:) AllTest(:,3)];%Matrix where the input variables are first then the output is the last column
[trainedClassifier,Fold_Gp0_AUC,Fold_Gp1_AUC]=mnl_SVM_CrossValidation(All_Data,nFolds);
%% Now apply the trained classifier onto the remaining data
FinalTestData=AllX(szAllTestSet+1:TotalN,:);
[yfit,scores]=trainedClassifier.predictFcn(FinalTestData);
%%Finally put it into the format requested
RowID=ArrayT(~AllGpIdx,1);
PredictedProb=scores(:,2);
OutputArray=[RowID PredictedProb];
OutputTable=array2table(OutputArray,'VariableNames',{'Row ID', 'Predicted Probability'});
writetable(OutputTable,'OutputTable.csv');
end
%% Subfunctions
function [OptFolds,OptLat]=mnl_AssessPLSbyCV(AllTest,nFolds,maxLatents)
szFolds=size(nFolds,2);
figure('Name','Cross Validation')
for i=1:szFolds
    [~,~,~,~,~,pctVar,PLSmseP]=plsregress(AllTest(:,4:303),AllTest(:,3),maxLatents,'CV',nFolds(i));
    legname{i}=sprintf('%d%s',nFolds(i),' Folds');
    subplot(1,2,1)
    plot(1:maxLatents,cumsum(100*pctVar(2,:)));
    hold on    
    subplot(1,2,2)
    plot(1:maxLatents+1,PLSmseP(2,:))
    hold on    
    %Store msePs to evaluate the optimum number of latents
    [M,I]=min(PLSmseP(2,:));
    %Summary Optimum
    OptimumLatents(i,1)=M;
    OptimumLatents(i,2)=I;
end
subplot(1,2,1)
xlim([0 maxLatents])
ylim([0 100])
title('Variance Explained')
xlabel('Number of Latents')
ylabel('Percent of Variance Explained in Y')
subplot(1,2,2)
title('MSE')
xlabel('Number of Latents')
ylabel('MSE')
xlim([0 maxLatents])
legend(legname)
%Now Determine the optimum number of latents and folds
OptLat=round(mean(OptimumLatents(:,2)));
[~,FoldPos]=min(OptimumLatents(:,1));
OptFolds=nFolds(FoldPos);
end
function [Gp0_AUC_plsDA,Gp1_AUC_plsDA]=mnl_AssessPLSbyDA(AllTest,OptFolds,OptLat)
szAll=size(AllTest,1);
c=cvpartition(szAll,'KFold',OptFolds);
cN=0;
figure('Name','ROC curves for both groups')
%Pre-allocation
for i=1:OptFolds %Find the biggest test size
    idxTest=test(c,i);
    TestSize(i)=sum(idxTest);
end
MxTestSize=max(TestSize);
Fold_Gp0_TPR=nan(OptFolds,MxTestSize+1);
Fold_Gp0_FPR=nan(OptFolds,MxTestSize+1);
Fold_Gp0_Thresh=nan(OptFolds,MxTestSize+1);
Fold_Gp0_AUC=nan(OptFolds,1);
Fold_Gp1_TPR=nan(OptFolds,MxTestSize+1);
Fold_Gp1_FPR=nan(OptFolds,MxTestSize+1);
Fold_Gp1_Thresh=nan(OptFolds,MxTestSize+1);
Fold_Gp1_AUC=nan(OptFolds,1);
%Evaluate each fold to be thorough
for i=1:OptFolds
    %Find out which belong to the test and the training set
    idxTrain=training(c,i);
    idxTest=test(c,i);
    TrainGroup=AllTest(idxTrain,:);
    TestGroup=AllTest(idxTest,:);
    TestSize=sum(idxTest);
    %Make sure the test set has values in both groups
    chk=unique(TestGroup(:,3));
    szC=size(chk,1);
    
    cN=cN+1;
    %Now perform the pls
    TrainGroupId=TrainGroup(:,3);
    [XL,YL,XS,YS,beta,PCTVAR,MSE,stats]=plsregress(TrainGroup(:,4:303),TrainGroupId,OptLat);
    %Apply to the test group
    x=[ones(TestSize,1) TestGroup(:,4:303)];
    Ypred=x*beta;
    %Find the values for the ROCs
    [Gp0_FPR,Gp0_TPR,Gp0_Thresh,Gp0_AUC] = perfcurve(TestGroup(:,3),Ypred,'0');
    [Gp1_FPR,Gp1_TPR,Gp1_Thresh,Gp1_AUC] = perfcurve(TestGroup(:,3),Ypred,'1');
    %Store all variables
    Fold_Gp0_TPR(cN,1:TestSize+1)=Gp0_TPR';
    Fold_Gp0_FPR(cN,1:TestSize+1)=Gp0_FPR';
    Fold_Gp0_Thresh(cN,1:TestSize+1)=Gp0_Thresh';
    Fold_Gp0_AUC(cN)=Gp0_AUC;
    Fold_Gp1_TPR(cN,1:TestSize+1)=Gp1_TPR';
    Fold_Gp1_FPR(cN,1:TestSize+1)=Gp1_FPR';
    Fold_Gp1_Thresh(cN,1:TestSize+1)=Gp1_Thresh';
    Fold_Gp1_AUC(cN)=Gp1_AUC;
    %Plot the single fold
    plot(Gp1_FPR,Gp1_TPR)
    hold on
    legname1{i}=sprintf('%s%d%s%s%s','Fold ',i,' - ',num2str(round(Gp1_AUC,3)),' AUC');
end
%Calculate the mean for group 0
%Have to interpolate along the thresholds
Gp0_minT=min(Fold_Gp0_Thresh(:),[],'all','omitnan');
Gp0_maxT=max(Fold_Gp0_Thresh(:),[],'all','omitnan');
Gp1_minT=min(Fold_Gp1_Thresh(:),[],'all','omitnan');
Gp1_maxT=max(Fold_Gp1_Thresh(:),[],'all','omitnan');
Gp0_ThreshRange=fliplr(Gp0_minT:0.01:Gp0_maxT);
Gp1_ThreshRange=fliplr(Gp1_minT:0.01:Gp1_maxT);
%Now Pre-allocate
szGp0=size(Gp0_ThreshRange,2);
szGp1=size(Gp1_ThreshRange,2);
%Put it as ones, because the maximum edges will be by default 1
nFold_Gp0_TPR=ones(OptFolds,szGp0);
nFold_Gp0_FPR=ones(OptFolds,szGp0);
nFold_Gp1_TPR=ones(OptFolds,szGp1);
nFold_Gp1_FPR=ones(OptFolds,szGp1);

for i=1:OptFolds
    %Group 0
    nPoints_Gp0=size(Fold_Gp0_Thresh(i,:),2);
    for j=1:nPoints_Gp0
        if j==1
            idx=Gp0_ThreshRange>=Fold_Gp0_Thresh(i,j);
        elseif j==nPoints_Gp0
            idx=Gp0_ThreshRange<Fold_Gp0_Thresh(i,j-1);
        else
            idx=Gp0_ThreshRange>=Fold_Gp0_Thresh(i,j) & Gp0_ThreshRange<Fold_Gp0_Thresh(i,j-1);
        end
        nFold_Gp0_TPR(i,idx)=Fold_Gp0_TPR(i,j);
        nFold_Gp0_FPR(i,idx)=Fold_Gp0_FPR(i,j);
    end
    %Now group 1
    nPoints_Gp1=size(Fold_Gp1_Thresh(i,:),2);
    for j=1:nPoints_Gp1
        if j==1
            idx=Gp1_ThreshRange>=Fold_Gp1_Thresh(i,j);
        elseif j==nPoints_Gp1
            idx=Gp1_ThreshRange<Fold_Gp1_Thresh(i,j-1);
        else
            idx=Gp1_ThreshRange>=Fold_Gp1_Thresh(i,j) & Gp1_ThreshRange<Fold_Gp1_Thresh(i,j-1);
        end
        nFold_Gp1_TPR(i,idx)=Fold_Gp1_TPR(i,j);
        nFold_Gp1_FPR(i,idx)=Fold_Gp1_FPR(i,j);
    end
end
mFold_Gp0_FPR=mean(nFold_Gp0_FPR,'omitnan');
mFold_Gp0_TPR=mean(nFold_Gp0_TPR,'omitnan');
Gp0_AUC_plsDA=mean(Fold_Gp0_AUC);
%Calculate the mean for group 1
mFold_Gp1_FPR=mean(nFold_Gp1_FPR,'omitnan');
mFold_Gp1_TPR=mean(nFold_Gp1_TPR,'omitnan');
Gp1_AUC_plsDA=mean(Fold_Gp1_AUC);
legname1{OptFolds+1}=sprintf('%s%s%s','Mean - ',num2str(round(Gp1_AUC_plsDA,3)),' AUC');

plot(mFold_Gp1_FPR,mFold_Gp1_TPR,'k','LineWidth',2)
axis equal
xlim([0 1])
ylim([0 1])
title('Group 1 - ROC')
legend(legname1)
end
function [trainedClassifier,Fold_Gp0_AUC,Fold_Gp1_AUC,Fold_Gp0_OptPoint,Fold_Gp1_OptPoint]=mnl_SVM_CrossValidation(trainingData,nKfolds)
%NB Decided not to do a separate fold for the PLS regression but rather use
%the full regression data. I'm cheating a bit here and using the in built
%MATLAB code generator. This also assumes the inputs are 3D as decided by
%the PLS-CV
%
% Inputs
% All_Data - Each variable is a column, with the last column being the
% response
%
% Outputs
% TrainedClassifier
% ValidationAccuracy
%% Initial defining stage
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4'});
predictorNames={'column_1', 'column_2', 'column_3'};
predictors=inputTable(:, predictorNames);
response = inputTable.column_4;
isCategoricalPredictor = [false, false, false];
%% Then train the classier on the full data
classificationSVM=fitcsvm(predictors,response, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
rng(1); % For reproducibility
classificationSVM=fitPosterior(classificationSVM); %Done to get probabilities
% Create the result struct with predict function
predictorExtractionFcn=@(x) array2table (x,'VariableNames', predictorNames);
svmPredictFcn= @(x) predict(classificationSVM,x);
trainedClassifier.predictFcn=@(x) svmPredictFcn(predictorExtractionFcn(x));
trainedClassifier.ClassificationSVM = classificationSVM;


%% Evaluate the classifier using k-folds
szAll=size(trainingData,1);
cN=0;
c=cvpartition(szAll,'KFold',nKfolds);
%Pre-allocation
for i=1:nKfolds %Find the biggest test size
    idxTest=test(c,i);
    TestSize(i)=sum(idxTest);
end
MxTestSize=max(TestSize);
Fold_Gp0_TPR=nan(nKfolds,MxTestSize+1);
Fold_Gp0_FPR=nan(nKfolds,MxTestSize+1);
Fold_Gp0_Thresh=nan(nKfolds,MxTestSize+1);
Fold_Gp0_AUC=nan(nKfolds,1);
Fold_Gp0_OptPoint=nan(nKfolds,1);
Fold_Gp1_TPR=nan(nKfolds,MxTestSize+1);
Fold_Gp1_FPR=nan(nKfolds,MxTestSize+1);
Fold_Gp1_Thresh=nan(nKfolds,MxTestSize+1);
Fold_Gp1_AUC=nan(nKfolds,1);
Fold_Gp1_OptPoint=nan(nKfolds,1);
%Now evaluate each fold
figure('Name','SVM evaluated by K folds')
for i=1:nKfolds
    cN=cN+1;
    %Find out which belong to the test and the training set
    idxTrain=training(c,i);
    idxTest=test(c,i);
    TrainInput=trainingData(idxTrain,1:3);
    TrainOutput=trainingData(idxTrain,4);
    TestInput=trainingData(idxTest,1:3);
    TestGroundOutput=trainingData(idxTest,4);
    TestSize=sum(idxTest);
    %Run the SVM on the fold - same settings as above
    classificationSVM_Fold=fitcsvm(TrainInput,TrainOutput, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    rng(1); % For reproducibility
    classificationSVM_Fold=fitPosterior(classificationSVM_Fold);
    
    [~,Score]=predict(classificationSVM_Fold,TestInput);
    [Gp0_FPR,Gp0_TPR,Gp0_Thresh,Gp0_AUC,Gp0_OptPoint]=perfcurve(TestGroundOutput,Score(:,1),'0');
    [Gp1_FPR,Gp1_TPR,Gp1_Thresh,Gp1_AUC,Gp1_OptPoint]=perfcurve(TestGroundOutput,Score(:,2),'1');
    %Store all variables
    Fold_Gp0_TPR(cN,1:TestSize+1)=Gp0_TPR';
    Fold_Gp0_FPR(cN,1:TestSize+1)=Gp0_FPR';
    Fold_Gp0_Thresh(cN,1:TestSize+1)=Gp0_Thresh';
    Fold_Gp0_AUC(cN)=Gp0_AUC;
    Fold_Gp0_OptPoint(cN)=Gp0_OptPoint(1);
    Fold_Gp1_TPR(cN,1:TestSize+1)=Gp1_TPR';
    Fold_Gp1_FPR(cN,1:TestSize+1)=Gp1_FPR';
    Fold_Gp1_Thresh(cN,1:TestSize+1)=Gp1_Thresh';
    Fold_Gp1_AUC(cN)=Gp1_AUC;
    Fold_Gp1_OptPoint(cN)=Gp1_OptPoint(1);
    %Plot the single fold
    plot(Gp1_FPR,Gp1_TPR)
    hold on
    legname1{i}=sprintf('%s%d%s%s%s','Fold ',i,' - ',num2str(round(Gp1_AUC,3)),' AUC');
end
%Calculate the mean for group 0
%Have to interpolate along the thresholds
Gp0_ThreshRange=fliplr(0:0.01:1);
Gp1_ThreshRange=fliplr(0:0.01:1);
%Now Pre-allocate
szGp0=size(Gp0_ThreshRange,2);
szGp1=size(Gp1_ThreshRange,2);
%Put it as ones, because the maximum edges will be by default 1
nFold_Gp0_TPR=ones(nKfolds,szGp0+1);
nFold_Gp0_FPR=ones(nKfolds,szGp0+1);
nFold_Gp1_TPR=ones(nKfolds,szGp1+1);
nFold_Gp1_FPR=ones(nKfolds,szGp1+1);

for i=1:nKfolds
    %Group 0
    nPoints_Gp0=size(Fold_Gp0_Thresh(i,:),2);
    for j=1:nPoints_Gp0
        if j==1
            idx=Gp0_ThreshRange>=Fold_Gp0_Thresh(i,j);
        elseif j==nPoints_Gp0
            idx=Gp0_ThreshRange<Fold_Gp0_Thresh(i,j-1);
        else
            idx=Gp0_ThreshRange>=Fold_Gp0_Thresh(i,j) & Gp0_ThreshRange<Fold_Gp0_Thresh(i,j-1);
        end
        nFold_Gp0_TPR(i,idx)=Fold_Gp0_TPR(i,j);
        nFold_Gp0_FPR(i,idx)=Fold_Gp0_FPR(i,j);
    end
    %Now group 1
    nPoints_Gp1=size(Fold_Gp1_Thresh(i,:),2);
    for j=1:nPoints_Gp1
        if j==1
            idx=Gp1_ThreshRange>=Fold_Gp1_Thresh(i,j);
        elseif j==nPoints_Gp1
            idx=Gp1_ThreshRange<Fold_Gp1_Thresh(i,j-1);
        else
            idx=Gp1_ThreshRange>=Fold_Gp1_Thresh(i,j) & Gp1_ThreshRange<Fold_Gp1_Thresh(i,j-1);
        end
        nFold_Gp1_TPR(i,idx)=Fold_Gp1_TPR(i,j);
        nFold_Gp1_FPR(i,idx)=Fold_Gp1_FPR(i,j);
    end
end

mFold_Gp0_FPR=mean(nFold_Gp0_FPR,'omitnan');
mFold_Gp0_TPR=mean(nFold_Gp0_TPR,'omitnan');
Gp0_AUC=mean(Fold_Gp0_AUC);
%Calculate the mean for group 1
mFold_Gp1_FPR=mean(nFold_Gp1_FPR,'omitnan');
mFold_Gp1_TPR=mean(nFold_Gp1_TPR,'omitnan');
Gp1_AUC=mean(Fold_Gp1_AUC);
legname1{nKfolds+1}=sprintf('%s%s%s','Mean - ',num2str(round(Gp1_AUC,3)),' AUC');
plot(mFold_Gp1_FPR,mFold_Gp1_TPR,'k','LineWidth',2)
axis equal
xlim([0 1])
ylim([0 1])
title('Group 1 - ROC')
legend(legname1)
end
