#####
# AUTHOR: Rama Hruday Bandaru
# FILENAME: main.r
# SPECIFICATION: Understand the use of Python or R to run logistic regression, model evaluation, k-fold cross validation
# FOR: CS 5331 – Special Topics in Machine Learning and Information Security 003
####

install.packages("tidyverse")
install.packages("GGally")
install.packages("ROCR")  
library("ROCR")
library(readr)
library(ggplot2)
library(GGally)

# Q1 Read “datafile.csv” into a dataframe
data_file <- read_csv("datafile.csv")

# Q2 Plot each attribute against each other and view how the data are related to each other
##   using Pairs 4 attributes at a time
pairs( ~ a1 + a2 + a3 + a4, data = data_file, col = data_file$a8)
pairs( ~ a5 + a6 + a7 + a8, data = data_file, col= data_file$a8)


##  Try out “ggpairs(dataframe)” from the ggplot2 and GGally packages
ggpairs(data_file,
        columns = 1:4,
        upper = list(continuous = "points"))
ggpairs(data_file,
        columns = 5:8,
        upper = list(continuous = "points"))


# Q3 Figure out how many 1’s and 0’s are in the a8 attribute column. Note the imbalance
table(data_file$a8)


# Q4 Make a new dataframe by selecting randomly 150 of the rows where a8=1 and a8=0.
# 150 of the rows where a8=1
filtered_1 = data_file[data_file$a8 == 1,]
rows_1 = sample(nrow(filtered_1), size = 150, replace = FALSE,)
filtered_1_frame <- filtered_1[rows_1,]
filtered_1_frame
# 150 of the rows where a8=0
filtered_0 = data_file[data_file$a8 == 0,]
rows_0 = sample(nrow(filtered_0), size = 150, replace = FALSE,)
filtered_0_frame <- filtered_0[rows_0,]
filtered_0_frame
# combine or bind both the 150 rows of values
new_data_frame = rbind(filtered_0_frame, filtered_1_frame)




####
# NAME: generateTrainTestSets
# PARAMETERS: train_percent (percentage of training set),data_frame (the data frame)
# PURPOSE: The function returns a list  of train and test subsets of data_frame,
# train percent is the percentage of the traing data
# PRECONDITION: data frame must be intialised and sent through the function
# POSTCONDITION: No variable change, the function returns a new list
####
generateTrainTestSets <- function(train_percent,data_frame) {
  size = nrow(data_frame)
  sample_size = train_percent * size / 100
  tr_row = sample(size, sample_size, replace = FALSE)
  train = data_frame[tr_row,]
  test = data_frame[-tr_row,]
  combine <- list("train" = train, "test" = test)
  return (combine)
}

# Q5 From the new dataframe, make a training and a test set where =>
# Seventy Thirty Train-Test
SeventyThirtySet = generateTrainTestSets(70,new_data_frame)

# Sixty Forty Train-Test
SixtyFortySet = generateTrainTestSets(60,new_data_frame)




####
# NAME: analyseModel
# PARAMETERS: test set and logistic model generated
# PURPOSE: The function prints out Confusion matrix,Accuracy and Area under PR
# PRECONDITION: No changes any global varibales
# POSTCONDITION: No variable change
####
analyseModel <- function(test, model) {
  test$Predict = predict(model, newdata = test, type = "response")
  test$Check = (ifelse(test$Predict > 0.5, 1, 0))
  
  cm = table(test$a8, test$Check)[2:1, 2:1]
  TN = cm[1, 1]
  TP = cm[2, 2]
  FP = cm[1, 2]
  FN = cm[2, 1]
  accuracy  = (TP + TN) / (TP + TN + FP + FN)
  
  #plotting the ROC
  pred = prediction(test$Check, test$a8)
  perf = performance(pred, "prec", "rec")
  plot(perf)
  roc = performance(pred, "tpr", "fpr")
  plot (roc, lwd = 2)
  abline(a = 0, b = 1)
  
  #plotting the precision recall curve
  pr_rec = performance(pred, "prec", "rec")
  plot(
    pr_rec,
    avg = "threshold",
    colorize = TRUE,
    lwd = 3,
    main = "... Precision/Recall graphs ..."
  )
  plot(pr_rec,
       lty = 3,
       col = "grey78",
       add = TRUE)
  
  #calculate
  aucpr = performance(pred, measure = "aucpr")
  
  print(paste("Confusion matrix: ", cm))
  print(paste("Accuracy: ", accuracy))
  print(paste("Area under PR: ", aucpr@y.values))
  print("### End of model ###")
  
}
####
# NAME: build5LogisticRegressionAndPrintAccuracy
# PARAMETERS: data_frame (the data frame)
# PURPOSE: The function prints out accurancy for follwoing attribute set
# a8 (a1,a2,a3,a4,) & a8 (a1,a2,a3) & a8 (a2,a3,a4) & a8 (a4,a5,a6) & a8 (a5,a6,a7)
# PRECONDITION: function only for specific dataset "datafile.csv"
# POSTCONDITION: No variable change, the dataSet manipulated is local to the function
####
build5LogisticRegressionAndPrintAccuracy <- function(data) {
  dataSet = data
  data_1_model <-
    glm(
      formula = a8 ~ a1 + a2 + a3 + a4 + a6 + a6 + a7,
      data = dataSet$train,
      family = binomial
    )
  analyseModel(dataSet$test, data_1_model)
  
  
  
  data_2_model <-
    glm(
      formula = a8 ~ a1 + a2 + a3 ,
      data = data$train,
      family = binomial
    )
  analyseModel(dataSet$test, data_2_model)
  
  
  
  data_3_model <-
    glm(
      formula = a8 ~ a2 + a3 + a4,
      data = data$train,
      family = binomial
    )
  analyseModel(dataSet$test, data_3_model)
  
  
  
  data_4_model <-
    glm(
      formula = a8 ~  a4 + a5 + a6,
      data = data$train,
      family = binomial
    )
  analyseModel(dataSet$test, data_4_model)
  
  
  
  data_5_model <-
    glm(
      formula = a8 ~ a5 + a6 + a7,
      data = data$train,
      family = binomial
    )
  analyseModel(dataSet$test, data_5_model)
  
}
## Q6 Build five logistic regression models on the training set using all attributes, a1 through a7 with a8, and 
## four different subsets of the attributes, such as using a1, a2, and a3 with a8

# Running all the five attribute sets with 70-30 train test set
build5LogisticRegressionAndPrintAccuracy(SeventyThirtySet)

# Running all the five attribute sets with 60-40 train test set
build5LogisticRegressionAndPrintAccuracy(SixtyFortySet)

# By observing the accuracy printed Accuracy with " a1 through a7 with a8 " 
# has best accuracy and more area under PR curve

## Q7 With the logistic regression model that you consider the best, 
# run the model on the entire new 
# dataframe, look at the confusion matrix, and note the accuracy
data_1_model <-
  glm(
    formula = a8 ~ a1 + a2 + a3 + a4 + a6 + a6 + a7,
    data = SeventyThirtySet$train,
    family = binomial
  )
analyseModel(new_data_frame, data_1_model)
# below is the Accuracy & Confusion matrix when run over the new data frame with the best model
# Accuracy:  0.736666666666667
# Area under PR:  0.794001157279535
# Confusion matrix:  [104, 33 ]
#                    [46, 117 ]

## Q8 Generate the ROC curve with the results from the entire new dataset.
## Q9 Generate the PR curve with the results from the entire new dataset. 
# THe below functions plots out both PR and ROC curve
analyseModel(new_data_frame, data_1_model)

## Q10 Using k-fold cross-validation, run logistic regression on the entire new dataframe with 2 different fold 
# sizes. Note the confusion tables, accuracies, and average accuracy.

####
# NAME: performKFoldValidationAndPrintAccuracy
# PARAMETERS: nFolds (number of folds/ sample-subset) ,data (the orginal data frame) 
# PURPOSE: The function performs k-fold cross validation and prints out average accuracy
# PRECONDITION: No variable change
# POSTCONDITION: No variable change
####
performKFoldValidationAndPrintAccuracy <- function(nfolds, data) {
  #Randomly shuffle the data
  shuffle <- data[sample(nrow(data)), ]
  #Create folds (makes a vector marking each fold, such as 1 1 1 2 2 2 3 3 3, for 3 folds)
  folds <-
    cut(seq(1, nrow(shuffle)), breaks = nfolds, labels = FALSE)
  #Perform 10 fold cross validation
  average_accuracy = 0.0
  for (i in 1:nfolds) {
    #Choose 1 fold and separate the testing and training data
    testIndexes <- which(folds == i, arr.ind = TRUE)
    testData <- shuffle[testIndexes,]
    trainData <- shuffle[-testIndexes,]
    #Build the model using logistic regression
    tshirtmod = glm(
      formula =  a8 ~ a1 + a2 + a3 + a4 + a6 + a6 + a7,
      data = trainData,
      family = binomial
    )
    testData$Predict = predict(tshirtmod, newdata = testData, type = "response")
    testData$Check = (ifelse(testData$Predict > 0.5, 1, 0))
    #make a confusion table (which might only have 1 column)
    x = table(testData$a8, testData$Check)
    if (ncol(x) == 1) {
      if (colnames(x)[1] == "0") {
        x = cbind(x, "1" = c(0, 0))
      } else {
        x = cbind("0" = c(0, 0), x)
      }
    }
    x = x[2:1, 2:1]
    accuracy = (x[1, 1] + x[2, 2]) / nrow(testData)
    average_accuracy = average_accuracy + accuracy
  }
  print(x)
  print(paste("Average Accuracy: ", average_accuracy / nfolds))
}

performKFoldValidationAndPrintAccuracy(3, new_data_frame)

performKFoldValidationAndPrintAccuracy(14, new_data_frame)

