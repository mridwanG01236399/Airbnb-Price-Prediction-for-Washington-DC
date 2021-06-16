library(tidyverse)
library(caret)

# 1. Preparation 
##########################################################################################################################
## 1.1. Import the data
##########################################################################################################################

# read the csv file
data <- read.csv("D:/George Mason University/4. Fall 2020/Applied Machine Learning (AIT 736)/Project/cleaning_data.csv")

# drop the "property_type"
data <- subset(data, select = -c(property_type))

# prepare the X values as the predictors
X <- subset(data, select = -c(id, price))

# prepare the y value as the target variable
y <- data$price

# data dimention of X
dim(X)

cat(dim(X)[2]," predictor are selected in this project:\n")
paste(names(X), collapse=', ' )

##########################################################################################################################
## 1.2. Categorical data feature engineering
##########################################################################################################################

# create a full set of dummy variables
dmy <- dummyVars(" ~ .", data = X, fullRank = T)

# transform to data frame
X <- data.frame(predict(dmy, newdata = X))

# new datad dimention after transformation
dim(X)

##########################################################################################################################
## 1.3. Missing Values & 0 Variance
##########################################################################################################################

# check the missing value
sum(is.na(X))

# check the near zero variance predictors
nearZeroVar(X,freqCut = 100/0)

# identify the predictors which have the near zero variance
names(X)[nearZeroVar(X,freqCut = 100/0)]

# drop the predictor which have the neear zero variance
X <- X[,-25]

# the new data dimention
dim(X)

##########################################################################################################################
## 1.4. Split the data
##########################################################################################################################

#set the seed for the split
set.seed(123)

# split the index  75 and 25 percent respectively for the train and test set
train_ind <- sample(seq_len(nrow(X)), size = 0.75*nrow(X))

# assign the train data of X based on the index split (train predictors)
trainX <- X[train_ind, ]

# assign the train data of y based on the index split (train target variable)
trainY <- y[train_ind]

# assign the test data of X based on the index split (test predictors)
testX <- X[-train_ind, ]

# assign the test data of y based on the index split (test target variable)
testY <- y[-train_ind]

# print the number of instances in the train set
cat("Number of instances in training set:",length(trainY),"\n")

# print the number of instances in the test set
cat("Number of instances in test set:",length(testY))

##########################################################################################################################
## 1.5. PCA Feature Reduction
##########################################################################################################################

# Apply PCA
PCA <- prcomp(trainX,center = TRUE, scale = TRUE)

PCA_Result <- data.frame(summary(PCA)[["importance"]]) 

PCA_Result <- as.data.frame(t(as.matrix(PCA_Result)))

PCA_Result$Eigenvalues <- PCA[["sdev"]]^2
PCA_Result
# The eigenvalues of the covariance/correlation matrix
# PCA[["sdev"]]^2
# The standard deviations of the principal components/ the square roots of the eigenvalues
# PCA[["sdev"]]
# The eigenvector of each PC
# PCA[["rotation"]] 

# The index of the PC of which Cumulative Proportion greater than 0.8
PC80 <- min(which((PCA_Result$`Cumulative Proportion`>0.8) == TRUE))
PCA_Result$`Cumulative Proportion`[PC80]

# The index of the PC of which Cumulative Proportion greater than 0.9
PC90 <- min(which((PCA_Result$`Cumulative Proportion`>0.9) == TRUE))
PCA_Result$`Cumulative Proportion`[PC90]

## Cumulative Proportion of Variance Plot (stair steps)
plot(PCA_Result$`Cumulative Proportion`, ylim=c(0,1.1),
     xlab="Component", ylab="Cumulative Proportion of Variance", 
     type="s", main="Cumulative Proportion of Variance")
points(PC80, PCA_Result$`Cumulative Proportion`[PC80],pch=19,col="red")
text(PC80, PCA_Result$`Cumulative Proportion`[PC80], labels=PC80,cex= 0.8,pos=3)
points(PC90, PCA_Result$`Cumulative Proportion`[PC90],pch=19,col="blue")
text(PC90, PCA_Result$`Cumulative Proportion`[PC90], labels=PC90,cex= 0.8,pos=3)

## Proportion of Variance Plot
plot(PCA_Result$`Proportion of Variance`, 
     xlab="Component", ylab="Proportion of Variance", 
     type="b", main="Proportion of Variance")
points(PC80, PCA_Result$`Proportion of Variance`[PC80],pch=19,col="red")
text(PC80, PCA_Result$`Proportion of Variance`[PC80], labels=PC80,cex= 0.8,pos=3)
points(PC90, PCA_Result$`Proportion of Variance`[PC90],pch=19,col="blue")
text(PC90, PCA_Result$`Proportion of Variance`[PC90], labels=PC90,cex= 0.8,pos=3)

## Result: only use the first 25 PCs with a Cumulative Proportion greater than 0.9
trainX_PCA <- data.frame(predict(PCA, newdata = trainX)[,1:PC90])
testX_PCA <- data.frame(predict(PCA, newdata = testX)[,1:PC90])

# data dimention for the first train X
dim(trainX)

# data dimention for the train X after the PCA is applied
dim(trainX_PCA)

##########################################################################################################################
## 1.6. PCA Feature Reduction
##########################################################################################################################

## To filter on correlations, we first get the correlation matrix for the predictor set
library(corrplot)

# build the correlation plot for the original train predictors
segCorr <- cor(trainX)
corrplot(segCorr,tl.cex = .4)

# build the correlation plot for the PCA train predictors
segCorr_PCA <- cor(trainX_PCA)
corrplot(segCorr_PCA)

##########################################################################################################################
# RPART
##########################################################################################################################
# Fit the model with Original data Transfomation
##########################################################################################################################
## Single trees
library(rpart)
library(party)
library(partykit)

ctrl <- trainControl(method = "cv", number = 5)

set.seed(100)
## rpart2 is used to tune max depth 
rpartTune1 <- train(x = trainX, 
                    y = trainY, 
                    method = "rpart2",
                    tuneLength = 20, 
                    trControl = ctrl)
rpartTune1

FinalTree1 = rpartTune1$finalModel

rpartTree1 = as.party(FinalTree1)
#dev.new()
plot(rpartTree1)

# Plot of Model Tuning
#plot(rpartTune1, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")
plot(rpartTune1)

rpart_prediction1 = predict(rpartTune1, n.trees = 100, newdata = testX)

rpartPR1 = postResample(pred=rpart_prediction1, obs=testY)
rpartPR1

##########################################################################################################################
# Fit the model with PCA data Transfomation
##########################################################################################################################

set.seed(100)
## rpart2 is used to tune max depth 
rpartTune2 <- train(x = trainX_PCA, 
                    y = trainY, 
                    method = "rpart2",
                    tuneLength = 20, 
                    trControl = ctrl)
rpartTune2

FinalTree2 = rpartTune2$finalModel

rpartTree2 = as.party(FinalTree2)
#dev.new()
plot(rpartTree2)

# Plot of Model Tuning
#plot(rpartTune2, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")
plot(rpartTune2)

rpart_prediction2 = predict(rpartTune2, n.trees = 100, newdata = testX_PCA)

rpartPR2 = postResample(pred=rpart_prediction2, obs=testY)
rpartPR2

##########################################################################################################################
# RANDOM FOREST
##########################################################################################################################
# Fit the model with Original data Transfomation
##########################################################################################################################
library(randomForest)

control <- trainControl(method="cv", number=5)
set.seed(100)
mtryValues <- c(7, 12, 15, 17, 19, 21, 25)
rf_default <- train(x = trainX,
                    y = trainY,
                    method="rf", 
                    ntree = 500,
                    tuneGrid=data.frame(mtry = mtryValues), 
                    trControl=control)
rf_default

plot(varImp(rf_default, scale=FALSE), top = 10)

rf_default$finalModel

# Plot of Model Tuning
plot(rf_default)

pre_rf_default = predict(rf_default,newdata=testX)

## performance evaluation
rfPR1 = postResample(pred = pre_rf_default, obs=testY)
rfPR1


rf_pred <- data.frame(predict(rf_default, newdata = testX))
rf_pred <- rf_pred %>%
  rename(Prediction = names(rf_pred)) %>%
  bind_cols(data.frame(Observation = testY)) %>%
  mutate(Residual = Observation-Prediction)

ggplot(data = rf_pred, mapping = aes(x = Prediction, y = Observation)) +
  geom_point(color = "#0072B2") +
  labs(x = "Predicted", y = "Observed", 
       title = "Observed vs. Predicted ")
ggplot(data = rf_pred, mapping = aes(x = Prediction, y = Residual)) +
  geom_point(color = "#0072B2") +
  labs(x = "Predicted", y = "Residuals", 
       title = "Residuals vs. Predicted ")

##########################################################################################################################
# Fit the model with PCA data Transfomation
#########################################################################################################################
control <- trainControl(method="cv", number=5)
set.seed(100)
mtryValues <- c(5, 7, 9, 11, 13, 16, 18)
rf_default2 <- train(x = trainX_PCA,
                     y = trainY,
                     method="rf", 
                     ntree = 500,
                     tuneGrid=data.frame(mtry = mtryValues), 
                     trControl=control)
rf_default2

plot(varImp(rf_default2, scale=FALSE), top = 10)

# Plot of Model Tuning
plot(rf_default2)

pre_rf_default2 = predict(rf_default2,newdata=testX_PCA)

## performance evaluation
rfPR2 = postResample(pred = pre_rf_default2, obs=testY)
rfPR2
