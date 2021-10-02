install.packages("tidyverse")
install.packages("GGally")
install.packages("ROCR")  #perform once on your system
library("ROCR")
library(readr)
library(ggplot2)
library(GGally)


data_file <- read_csv("datafile.csv")

data_file$a8 <- as.factor(data_file$a8)


# Plot each attribute against each other and view how the data are related to each other
# using Pairs 4 attributes at a time
pairs(~ a1 + a2 + a3 + a4, data = data_file, col = data_file$a8)
pairs(~ a5 + a6 + a7 + a8, data = data_file, data_file$a8)

# using gg pairs 4 attributes at a time
ggpairs(data_file,
        columns = 1:4,
        upper = list(continuous = "points"))
ggpairs(data_file,
        columns = 5:8,
        upper = list(continuous = "points"))


# Figure out how many 1’s and 0’s are in the a8 attribute column. Note the imbalance.
table(data_file$a8)


# Make a new dataframe by selecting randomly

# 150 of the rows where a8=1
filtered_1 = data_file[data_file$a8 == 1, ]
rows_1 = sample(nrow(filtered_1), size = 150, replace = FALSE, )
filtered_1_frame <- filtered_1[rows_1, ]
filtered_1_frame

# 150 of the rows where a8=0
filtered_0 = data_file[data_file$a8 == 0, ]
rows_0 = sample(nrow(filtered_0), size = 150, replace = FALSE, )
filtered_0_frame <- filtered_0[rows_0, ]
filtered_0_frame

new_data_frame = rbind(filtered_0_frame, filtered_1_frame)
size = nrow(new_data_frame)

generateRange <- function(percentage) {
  sample_size = percentage * size / 100
  tr_row = sample(size, sample_size, replace = FALSE)
  return (tr_row)
}

# Seventy Thirty Train-Test
tr_row_1 = generateRange(70)
train_1 = new_data_frame[tr_row_1, ]
test_1 = new_data_frame[-tr_row_1, ]
test_1

# Sixty Forty Train-Test
tr_row_2 = generateRange(60)
train_2 = new_data_frame[tr_row_2, ]
test_2 = new_data_frame[-tr_row_2, ]
test_2

printModel <- function(te,model) {
  te$Predict = predict(model, newdata = te, type = "response")
  te$Check = (ifelse(te$Predict > 0.5, 1, 0))
  summary(model)
  
  x_1 = table(te$a8, te$Check)[2:1, 2:1]
  TN =x_1[1,1]
  TP =x_1[2,2]
  FP =x_1[1,2]
  FN =x_1[2,1]
  accuracy_model  =(TP+TN)/(TP+TN+FP+FN)
  print(accuracy_model)
  pred = prediction(te$Check, te$a8)
  perf = performance(pred, "prec", "rec")
  plot(perf)
  roc = performance(pred, "tpr", "fpr")
  plot (roc, lwd = 2)
  abline(a = 0, b = 1)
}


data_1_model <-
  glm(
    formula = a8 ~ a1 + a2 + a3 + a4 + a6 + a6 + a7,
    data = train_1,
    family = binomial
  )
printModel(test_1,data_1_model)


data_2_model <-
  glm(
    formula = a8 ~ a1 + a2 + a3 ,
    data = train_1,
    family = binomial
  )
printModel(test_1,data_2_model)


data_3_model <-
  glm(
    formula = a8 ~ a2 + a3 + a4,
    data = train_1,
    family = binomial
  )
printModel(test_1,data_3_model)


data_4_model <-
  glm(
    formula = a8 ~  a4 + a5 +a6,
    data = train_1,
    family = binomial
  )
printModel(test_1,data_4_model)



data_5_model <-
  glm(
    formula = a8 ~ a5 + a6 +a7,
    data = train_1,
    family = binomial
  )
printModel(test_1,data_5_model)


