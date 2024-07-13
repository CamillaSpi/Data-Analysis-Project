################################################UTILS################################################
library(glmnet)
library(ISLR)
library(corrplot)
library(car)
library(psych)
library(plotmo)

#R SQUARED error metric -- Coefficient of Determination
RSQUARE = function(y_actual,y_predict){
  cor(y_actual,y_predict)^2}




################################################LOAD DATASET################################################
Dataset=read.csv("RegressionExam_8feb23.csv",header=T,na.strings ="?")
dim(Dataset)
Dataset=na.omit(Dataset)
#there are no not available values in fact the dimensions remain the same
dim(Dataset[])
head(Dataset) #first row of the dataset, all the variables are numeric one
names(Dataset)#names of the predictors and of the response




################################################DATA ANALYSIS################################################
pairs(Dataset) #plot pairs in pairs
corBo <- round(cor(Dataset), digits =2)#correlaion matrix
corrplot(corBo)#plot of the correlation matrix
#Observing the first row of the correlation matrix it is possible to notice the correlation level of each predictor respect to the response
#and it is possible to see that it is an high value only for a small number of the predictors, in particulare the higher ones are x23,x24,x25

#obtainig predictors and response
x = model.matrix(Y~., Dataset)[,-1] #predictors
y = Dataset$Y #response
dim(x)
length(y)

boxplot(x, horizontal=T, xlab="X",ylab="", main="Box-plot X") #boxplot for range interquartaile, and eventually outliers
hist(Dataset$Y) #Histogram of response




################################################TRAIN-TEST SPLIT OF THE DATASET################################################
#split 80% for training and 20% for validation as required by requirements
set.seed(2023)
train=sample(1:nrow(x), 0.8*nrow(x)) #obtaining the indices for the split
test=(-train)
y.test=y[test]

#dimensions of training and test set
length(train)
length(test)
length(y.test)
dim(x[test,])








################################################BASIC MULTIPLE LINEAR REGRESSION MODEL################################################
#split the Dataset in test and training one, using the indices calculated above
trainDataSet = Dataset[train,]
dim(trainDataSet)
testDataSet = Dataset[test,]
dim(testDataSet)

fit = lm(Y~., data=trainDataSet)# fit a linear model with all predictors
summary(fit) #many p-values have a very large value and so this means that probably an approach of variable selection is required
#in order to extract only the more significant predictors.
#The fact that the R-squared is to high means that the variance of the model have been almost fully explained. 
#But it is important to consider that increasing the number of predictors the value of R-squared tend to increase always.
#The value of F-statistic is very high, accordingly to that explained above.


vif(fit)#check for multicollinearity, as it is possible to see the values of Variance Inflation Factor, there are no values that exceeds much 5 or 10
#that indicates a problematic amount of collinearity, as it is possible to see also from correlation matrix. 
dev.new();corrplot.mixed(corBo,order='original',number.cex=1, upper="ellipse")
corBo <- round(cor(Dataset), digits =2)#correlaion matrix
corrplot(corBo)#plot of the correlation matrix

#evaluate the error on the test set
basic_reg_pred=predict(fit,newx=x[test,])
mse_basic_reg <- mean((basic_reg_pred-y.test)^2)#obtained value 260427747
print(mse_basic_reg) #as it is possible to see the mse is very large, but is it reasonable in according to all the above considerations, about the inadequacy
# of basic multiple linear regression to face with a problem as this analyzed

confint(fit)# confidence interval for coefficients, default is at 95%

dev.new();par(mfrow=c(2,2));plot(fit)
#Q-Q plot compares the quantile of the data to that of a normal distribution(default options) and how it is possible to see the plot is 
#is fairly aligned along the bisector. In particular it is a test of gaussianity because we are analyzing the residuals and for different evaluations we assume that gaussianity is satisfied.  
#If this does not happens it means or that the initial error is not gaussian(another assumption we generally do), or that our model introduced an error so large that the quantiles were wrong.
#From the bottom-left image it is possible to see that there are no high-leverage points, because there aren't points that exceeds cook regions.
#From the bottom-right image it is possible to see that there are some potencially outliers, but not relevant.
#From the first image it is possible to see that there are not worrying situations about collinearity.
#Because of the fact that the problem is quite high dimension, probably the basic regression problem is not a good approach.
#Because of this reason also more adequate types of approach have been tested.








################################################BASIC MULTIPLE LINEAR REGRESSION MODEL WITH SUBSET SELECTION THROUGH AIC AND BIC################################################
trainDataSet = Dataset[train,]
dim(trainDataSet)
testDataSet = Dataset[test,]
dim(testDataSet)

#perfrom a stepwise subset selection with an hybrid approach, using for evaluate the best model the adjusted statistical of BIC and AIC
startmod=lm(Y~1, data=trainDataSet)
startmod
scopmod=lm(Y~., data=trainDataSet)
scopmod

#VARIABLE SELECTION WITH AIC
newvarAIC = optmodAIC <- step(startmod,direction = "both",k=2, scope=formula(lm(Y~., data=trainDataSet))) #obtain coefficient 
length(newvarAIC$coefficients[-1])#11
preds = newvarAIC$coefficients[1]+(x[test,names(newvarAIC$coefficients)[-1]])%*%(newvarAIC$coefficients[-1]) #compute prediction with obtained coefficient
preds = t(preds)
#evaluate the error on the test set
mse_aic_reg <- mean((preds-y.test)^2) #obtained MSE 7281.48
print(mse_aic_reg) #as it is possible to see the MSE has a good improvement respect the basic OLS, because we perform a variable selection, that now became 11

print(newvarAIC$coefficients)#coefficient estimates

#coefficient manipulation to obtain the clue
for(v in newvarAIC$coefficients[order(names(newvarAIC$coefficients))]){
  print(intToUtf8(round(v/100)))
}

RSQUARE(t(preds),y.test) #NEW 0.9999458


#VARIABLE SELECTION WITH BIC
newvarBIC = optmodBIC <- step(startmod,direction = "both",k=log(dim(x)[2]), scope=formula(lm(Y~., data=trainDataSet))) #obtain coefficient 
length(newvarBIC$coefficients[-1])#9 

preds = newvarBIC$coefficients[1]+(x[test,names(newvarBIC$coefficients)[-1]])%*%(newvarBIC$coefficients[-1]) #compute prediction with obtained coefficient
preds = t(preds)
#evaluate the error on the test set
mse_bic_reg <- mean((preds-y.test)^2) #obtained value = 7154.536
print(mse_bic_reg)  #as it is possible to see the mse has a good improvement respect the basic OLS, because we perform a variable selection, that now became 9
#it is possible to see that with BIC we obtain a small values of regressors, because its penalty is more stringent respect to AIC

print(newvarBIC$coefficients)#coefficient estimates

#coefficient manipulation to obtain the clue
for(v in newvarBIC$coefficients[order(names(newvarBIC$coefficients))]){
  print(intToUtf8(round(v/100)))
}
RSQUARE(t(preds),y.test) #NEW 0.9999485







 
################################################RIDGE################################################
grid=10^seq(10,-2,length=100)#define range values in which try the values of lambda
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid,thresh=1e-12)#fit a generalized linear model
dev.new(); plot(ridge.mod, xvar = "lambda")#see the behavior of the coefficients for different values of lambda
set.seed (2022)
cv.out=cv.glmnet(x[train,],y[train],alpha=0,nfolds = 5,lambda=grid)#do k-fold cross-validation for glmnet
dev.new();plot(cv.out)#plot for different values of lambda, the value of the MSE. On the plot it is possible to see the value of lambda for the best MSE, and also the value 
#of lambda corresponding to the one-standard-error. On the top of the plot it is possible to see the numbers of regressors that remain used for different values of lambda.
#Obviusually in this case, because we are applying Ridge, no subset selection is performed, and so the number of regressors does not change, changing the values of lambda, 
#but they are only shrinkaged to zero.
bestlam=cv.out$lambda.min; bestlam; log(bestlam) #obtain the best lambda 
cv.out$lambda.1se #the value of lambda according to one standard error rule
log(cv.out$lambda.1se) 
ridge.pred=predict(ridge.mod,s=bestlam ,newx=x[test,]) # make prediction with min error of lambda 
mse_ridge <- mean((ridge.pred-y.test)^2); mse_ridge #compute the MSE on the test set #obtained MSE value = 9712.763

out=glmnet(x,y,alpha=0,lambda=grid)# Finally refit our ridge regression model on the full data set with the best lambda
values = predict(out,type="coefficients",s=bestlam)#prediction with the values of best lambda
# As expected, none of the coefficients are zero, because ridge regression does not perform variable selection
dev.new();plot(out,label = T, xvar = "lambda")
dev.new();plot_glmnet(out)#plot the coefficient paths of a glmnet model

#values of coefficient related to the weight of the penality term
dev.new();plot(out)

print(values@x)#coefficient estimates

#coefficient manipulation to obtain the clue
for(v in values@x){
  print(intToUtf8(round(v/100)))
}
RSQUARE(ridge.pred,y.test) #0.9999281
dim(ridge.pred)
y.test
#To perform cross validation different values of k have been tried, but in the script is left only the best one. The obtained results
#are anyway reported 
#k=3 -> 9712.763
#k=5 ->  9712.763
#k=7 -> 9712.763
#k=10 -> 9712.763
#Starting from this value it is possible to see that for values of k that are 3,5,7,10 there are no differences between the test mse.
#For all values of K there are no differences, but anyway the test MSE is quite large, this because it is obviusually a problem
#that require variable selection.
#Between this values we chose k=5. This because probably it is the right trade-off in terms 
#of not having very much data to fit the model at each iterations, even if having larger validation set at each iterations,
#and of having a very small validation set at each iterations, that means have less qualitative estimates.








################################################LASSO################################################
grid=10^seq(10,-2,length=100)#define range values in which try the values of lambda
lasso.mod = glmnet(x[train,], y[train], alpha=1, lambda=grid)#fit a generalized linear model
dev.new(); plot(lasso.mod, xvar = "lambda")#see the behavior of the coefficients for different values of lambda

# perform cross-validation
set.seed (2022)
cv.out=cv.glmnet(x[train,],y[train],alpha=1, nfolds = 3,lambda=grid)#do k-fold cross-validation for glmnet
dev.new();plot(cv.out)#plot for different values of lambda, the value of the MSE. On the plot it is possible to see the value of lambda for the best MSE, and also the value 
#of lambda corresponding to the one-standard-error. On the top of the plot it is possible to see the numbers of regressors that remain used for different values of lambda.
#Obviusually in this case, because we are applying Lasso, subset selection is performed, and so the number of regressors changes, changing the values of lambda, 
#so some of them became zero.

bestlam=cv.out$lambda.min; print(bestlam);print(log(bestlam))#obtain the best lambda 
lambdaTrue=cv.out$lambda.1se #the value of lambda according to one standard error rule
print(cv.out$lambda.1se)
print(log(cv.out$lambda.1se))

lasso.pred=predict(lasso.mod,s=bestlam ,newx=x[test,])# make prediction with min error of lambda 
mse_lasso <- mean((lasso.pred-y.test)^2); mse_lasso #best obtained 4478.796


out = glmnet(x, y, alpha=1, lambda=grid)# Finally refit our lasso regression model on the full data set with the best lambda
values = predict(out,s=bestlam,type="coefficients")#prediction with the values of best lambda
# As expected, some of the coefficients are zero, because lasso regression performs variable selection
dev.new();plot_glmnet(out)

#values of coefficient related to the weight of the penality term
dev.new();plot(out)

print(values@x)#coefficient estimates

#coefficient manipulation to obtain the clue
for(v in values@x){
  print(intToUtf8(round(v/100)))
}
RSQUARE(lasso.pred,y.test) # 0.9999576

#To perform cross validation different values of k have been tried, but in the script is left only the best one. The obtained results
#are anyway reported
#k=3 -> 4478.796
#k=5 ->  5992.016
#k=7 -> 5551.427
#k=10 -> 5551.427
#Starting from this results it is possible to see that the best value is k=3. This because probably it is the right trade-off in terms 
#of not having very much data to fit the model at each iterations, even if having larger validation set at each iterations,
#and of having a very small validation set at each iterations, that means have less qualitative estimates.








################################################ELASTIC NET################################################
grid=10^seq(10,-2,length=100)#define range values in which try the values of lambda

alpha_values=c(0,0.05, 0.25, 0.5, 0.75, 0.95,1) #define different values of alpha in order to try different configurations, in order to chose
#the best mixing of l1 and l2 penalty
i=1
mse_enet=rep(NA,7)#define an array to contain all the resulting MSE test errors
for(alp in alpha_values){
  print(alp)
  print(i)
  
  enet.mod = glmnet(x[train,], y[train], alpha=alp, lambda=grid)#fit a generalized linear model
  dev.new();
  plot_glmnet(enet.mod, xvar = "lambda")#see the behavior of the coefficients for different values of lambda
  
  # perform cross-validation
  set.seed (2022)
  cv.out=cv.glmnet(x[train,],y[train],alpha=alp,nfolds = 3,lambda=grid)#do k-fold cross-validation for glmnet
  dev.new();plot(cv.out)#plot for different values of lambda, the value of the MSE. On the plot it is possible to see the value of lambda for the best MSE, and also the value 
  #of lambda corresponding to the one-standard-error. On the top of the plot it is possible to see the numbers of regressors that remain used for different values of lambda.
  #Obviusually in this case, because we are applying ElasticNet that is a mix of Lasso and Ridge, subset selection is performed, and so the number of regressors changes,
  #changing the values of lambda, so some of them became zero.
  
  bestlam_enet=cv.out$lambda.min; print(bestlam_enet);print(log(bestlam_enet))#obtain the best lambda 
  print(cv.out$lambda.1se)#the value of lambda according to one standard error rule
  print(log(cv.out$lambda.1se))
  
  enet.pred=predict(enet.mod,s=bestlam_enet ,newx=x[test,])# make prediction with min error of lambda
  mse_enet[i] <- mean((enet.pred-y.test)^2) #each time save for the different value of alpha the value of the test MSE
  print(mse_enet)
  
  i=i+1
}

best_mse = min(mse_enet)# compute the min test MSE
print(best_mse)#best obtained 4478.796
min_index = which.min(mse_enet) # compute the index of the min MSE
print(min_index)
best_alp = alpha_values[min_index] #the values of alpha, between that explored, that minimizes the test MSE
print(best_alp)
out = glmnet(x, y, alpha=best_alp, lambda=grid)# Finally refit our ridge regression model on the full data set with the best lambda and the best alpha
values = predict(out,s=bestlam_enet,exact=T,type="coefficients")#prediction with the values of best lambda
# As expected, some of the coefficients are zero, because ElasticNet performs variable selection
dev.new();plot_glmnet(out)

#values of coefficient related to the weight of the penality term
dev.new();plot(out)

print(values@x)#coefficient estimates

#coefficient manipulation to obtain the clue
for(v in values@x){
  print(intToUtf8(round(v/100)))
}
RSQUARE(enet.pred,y.test) #0.9999604

#To perform cross validation different values of k have been tried, but in the script is left only the best one. The obtained results
#are anyway reported above for the different values of alpha
#k=3 -> 9755.636 9782.046 7659.451 5524.495 5132.915 4595.025 4478.796 , best_alpha = 1, num pred=10
#k=5 -> 9755.636 9782.046 7231.490 6348.191 6052.819 5690.369 5992.016 , best_alpha = 0.95, num pred=16
#k=7 -> 9755.636 9782.046 7231.490 5942.455 5637.127 5690.369 5551.427 , best_alpha = 1, num pred=16
#k=10 -> 9755.636 9782.046 7231.490 5942.455 5637.127 5690.369 5551.427 , best_alpha = 1, num pred=16
#Starting from this results it is possible to see that the best value is k=3. This because probably it is the right trade-off in terms 
#of not having very much data to fit the model at each iterations, even if having larger validation set at each iterations,
#and of having a very small validation set at each iterations, that means have less qualitative estimates.
#Starting from the obtained results it is possible to see that in general Lasso works better than ElasticNet, in fact Lasso corresponds to 
#set the value of alpha equal to 1. This higliths that in a situation like this, in which visibly only a small subset of predictors
#have an higher value of significance, Lasso approach performs better, because do variable selection. So Lasso performs better than Ridge because
#it is a situation in which variable selection is required, and in fact an higher value of alpha means to tend more towards Lasso than Ridge.
#The only exception is for k=5 for which, probably for the configurations of training and validation split, results that ElasticNet has better performance
#respect to Lasso.
#And in particular analyzing the values of test MSE varying alpha, for k=5, it is possible to see a U-shape, that also allow to obtain for alpha=0.95
#a more stringent selection.





