library(tidyverse)
library(caret)
library(glmnet)
data <- read.csv("cleaning_data.csv")
data <- subset(data, select = -c(id,property_type))
dmy <- dummyVars(" ~ .", data = data, fullRank = T)
data <- data.frame(predict(dmy, newdata = data))
set.seed(123) 

# creating training data as 75% of the dataset 
random_sample <- createDataPartition(data $ price,  
                                     p = 0.75, list = FALSE) 

# generating training dataset 
training_dataset  <- data[random_sample, ] 

# generating testing dataset 
testing_dataset <- data[-random_sample, ] 

# Building the model 
model <- lm(price ~., data = training_dataset) 

# predicting the target variable 
predictions <- predict(model, testing_dataset) 

# computing model performance metrics 
data.frame( R2 = R2(predictions, testing_dataset $ price), 
            RMSE = RMSE(predictions, testing_dataset $ price), 
            MAE = MAE(predictions, testing_dataset $ price)) 

#PCA
X <- subset(data, select = -c(price))
X <- data.frame(predict(dmy, newdata = X))
X <- X[,-25]
y <- data$price
set.seed(123)
train_ind <- sample(seq_len(nrow(X)), size = 0.75*nrow(X))
trainX <- X[train_ind, ]
trainY <- y[train_ind]
testX <- X[-train_ind, ]
testY <- y[-train_ind]
data_train<- cbind(trainX, trainY)
data_train1<- cbind(testX, testY)
PCA <- prcomp(trainX,center = TRUE, scale = TRUE)
PCA_Result <- data.frame(summary(PCA)[["importance"]]) 
PCA_Result <- as.data.frame(t(as.matrix(PCA_Result)))
PCA_Result$Eigenvalues <- PCA[["sdev"]]^2
PC90 <- min(which((PCA_Result$`Cumulative Proportion`>0.9) == TRUE))
PCA_Result$`Cumulative Proportion`[PC90]
data_pca <- data.frame(predict(PCA, newdata = data)[,1:PC90])
data_pca <- cbind(data_pca, y)
trainX_PCA <- data.frame(predict(PCA, newdata = trainX)[,1:PC90])
testX_PCA <- data.frame(predict(PCA, newdata = testX)[,1:PC90])
train_data <- cbind(trainX_PCA, trainY)
test_data<-cbind(testX_PCA, testY)
#use pca
model_pca <- lm(trainY ~., data = train_data) 

# printing model performance metrics 
# along with other details 
print(model_pca)

# predicting the target variable 
predictions <- predict(model_pca, test_data) 

# computing model performance metrics 
data.frame( R2 = R2(predictions, test_data $ testY), 
            RMSE = RMSE(predictions, test_data $ testY), 
            MAE = MAE(predictions,test_data $ testY )) 

#lasso
#train_x<- as.matrix(X)
#train_y<- y

#set.seed(123)
#fit = glmnet(train_x, train_y)
#plot(fit)
#print(fit)
#coef(fit,s=0.1)

#set.seed(123)
#cvfit = cv.glmnet(train_x, train_y)
#plot(cvfit)
#cvfit$lambda.min
#coef(cvfit, s = "lambda.min")

#lassoGrid <- expand.grid(alpha = 1, lambda = seq(0,0.1,length = 20))
#set.seed(123)
#ctrl<-trainControl(method="repeatedcv", repeats=5)
#lassoFit <- train(x=train_x, y=train_y, method='glmnet', trControl= ctrl, preProc = c("center", "scale"),tuneGrid=lassoGrid) 
#lassoFit
#varImp(lassoFit)
