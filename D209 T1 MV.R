#Import necessary libraries
library(readr) #import data
library(dplyr) #data manipulation
library(corrr) #heat map
library(ggplot2) #heat map
library(caret) #split data
library(class) #build KNN model
library(pROC) #plot ROC/AUC

#Load the data
med <- read_csv("WGU/D209_medical_raw.csv")

# Get summary of data
str(med)

#Rename survey response variables
colnames(med)[colnames(med) == 'Item1'] <- 'Timely_admission'   
colnames(med)[colnames(med) == 'Item2'] <- 'Timely_treatment'   
colnames(med)[colnames(med) == 'Item3'] <- 'Timely_visits'   
colnames(med)[colnames(med) == 'Item4'] <- 'Reliability'   
colnames(med)[colnames(med) == 'Item5'] <- 'Options'   
colnames(med)[colnames(med) == 'Item6'] <- 'Hours_of_treatment'   
colnames(med)[colnames(med) == 'Item7'] <- 'Courteous_staff'   
colnames(med)[colnames(med) == 'Item8'] <- 'Active_listening'   

#Assess for duplicates
sum(duplicated(med))

#Assess for nulls
sum(is.na(med))

#Assess for outliers with histograms for numeric variables
par(mar=c(1,1,1,1))

hist(med$Children)
hist(med$Age)
hist(med$Income)
hist(med$VitD_levels)
hist(med$Doc_visits)
hist(med$Full_meals_eaten)
hist(med$vitD_supp)
hist(med$Initial_days)
hist(med$TotalCharge)
hist(med$Additional_charges)
hist(med$Timely_treatment)
hist(med$Active_listening)

#Encode categorical dependent variable as numeric
med$ReAdmis[med$ReAdmis == 'Yes'] <- 1
med$ReAdmis[med$ReAdmis == 'No'] <- 0

#New data frame with KNN variables
med_reduced <- med[, c("ReAdmis", "Children", "Age", "Income", "VitD_levels", 
                     "Doc_visits", "Full_meals_eaten", "vitD_supp", 
                     "Initial_days", "TotalCharge", "Additional_charges", 
                     "Timely_treatment", "Active_listening")]

#Visualize correlation
med_reduced %>%   
  dplyr::select(where(is.numeric)) %>%   
  correlate() %>%   
  shave() %>%   
  rplot(print_cor = TRUE) +   
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Correlation between Additional charges and Age under 0.75 threshold, keep
#Correlation between Total charge & Initial days over 0.75, dropping Total charge
med_reduced <- subset(med_reduced, select = -c(TotalCharge))

#Confirm drop
str(med_reduced)

#Visualize correlation again
med_reduced %>%   
  dplyr::select(where(is.numeric)) %>%   
  correlate() %>%   
  shave() %>%   
  rplot(print_cor = TRUE) +   
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Normalize and scale the data, convert to numeric
normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

med_norm <- as.data.frame(sapply(med_reduced, function(x) if(is.numeric(x)) normalize(x) else x))
med_clean <- as.data.frame(lapply(med_norm, function(x) if(is.character(x)) as.numeric(x) else x))

#Confirm data set normalized
str(med_clean)
glimpse(med_clean)

#Export cleaned data set
write_csv(med_clean, "WGU/D209_T1_MV_clean.csv")

#Set seed to reproduce same split
set.seed(123)

#Create indices
train_index <- createDataPartition(med_clean$ReAdmis, p=0.8, list=FALSE)

#Create the training and test data sets
train_set <-med_clean[train_index, ]
test_set <- med_clean[-train_index, ]

#Confirm split
dim(train_set)
dim(test_set)

#Further split data (y as data frame so I can export to CSV)
x_train <- train_set[, -which(names(train_set) %in% "ReAdmis")]
y_train <- as.data.frame(train_set$ReAdmis)

x_test <- test_set[, -which(names(test_set) %in% "ReAdmis")]
y_test <- as.data.frame(test_set$ReAdmis)

#Export as training & testing data
write_csv(x_train, "WGU/D209_T1_MV_xtrain.csv")
write_csv(x_test, "WGU/D209_T1_MV_xtest.csv")
write_csv(y_train, "WGU/D209_T1_MV_ytrain.csv")
write_csv(y_test, "WGU/D209_T1_MV_ytest.csv")

# Convert y back to vector
y_train <- as.vector(unlist(y_train))
y_test <- as.vector(unlist(y_test))

# Calculate k-value
n <- nrow(train_set)
k <- sqrt(n)
print(k)

#Fit KNN models
knn_89 <- knn(train=x_train, test=x_test, cl=y_train, k=89, prob=TRUE)
knn_90 <- knn(train=x_train, test=x_test, cl=y_train, k=90, prob=TRUE)

#Confusion matrix
cfm_89 <- table(knn_89, y_test)
cfm_89
cfm_90 <- table(knn_90, y_test)
cfm_90

#Accuracy
acc_89 <- mean(y_test == knn_89)
acc_89
acc_90 <- mean(y_test == knn_90)
acc_90

#Create y_test_numeric for ROC/AUC
y_test_numeric <- ifelse(y_test == "1", 1, 0)

#Plot ROC/AUC for knn_89
probs_89 <- attr(knn_89, "prob")
readmit_prob_89 <- ifelse(knn_89 == "1", probs_89, 1-probs_89)
roc_89 <- roc(response=y_test_numeric, predictor=readmit_prob_89)
plot(roc_89, main="ROC Curve for knn_89")
auc(roc_89)

#Plot ROC/AUC for knn_90
probs_90 <- attr(knn_90, "prob")
readmit_prob_90 <- ifelse(knn_90 == "1", probs_90, 1-probs_90)
roc_90 <- roc(response=y_test_numeric, predictor=readmit_prob_90)
plot(roc_90, main="ROC Curve for knn_90")
auc(roc_90)
