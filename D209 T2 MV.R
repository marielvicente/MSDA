#Import necessary libraries
library(readr) #import data
library(dplyr) #data manipulation
library(corrr) #heat map
library(ggplot2) #heat map
library(caret) #split data
library(randomForest) #build random forest model
library(Metrics) #calculate RSME & MSE

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

#Assess for outliers with histograms for quantitative variables
par(mar=c(1,1,1,1))

hist(med$Initial_days)
hist(med$Children)
hist(med$Age)
hist(med$Income)
hist(med$VitD_levels)
hist(med$Doc_visits)
hist(med$Full_meals_eaten)
hist(med$TotalCharge)
hist(med$Additional_charges)
hist(med$Timely_treatment)
hist(med$Active_listening)

#Encode categorical variables (Complication risk & Asthma) as numeric
med$Complication_risk[med$Complication_risk == 'Low'] <- 1
med$Complication_risk[med$Complication_risk == 'Medium'] <- 2
med$Complication_risk[med$Complication_risk == 'High'] <- 3

med$Asthma[med$Asthma == 'Yes'] <- 1
med$Asthma[med$Asthma == 'No'] <- 0

#New data frame with random forest variables, as numeric
med_clean <- med %>%
  select(Initial_days, Children, Age, Income, VitD_levels,
          Doc_visits, Full_meals_eaten, TotalCharge, 
          Additional_charges, Timely_treatment, Active_listening,
          Complication_risk, Asthma) %>%
  mutate_at(vars(Initial_days, Children, Age, Income, VitD_levels,
                 Doc_visits, Full_meals_eaten, TotalCharge, 
                 Additional_charges, Timely_treatment, Active_listening,
                 Complication_risk, Asthma), as.numeric)

glimpse(med_clean)

#Visualize correlation
med_clean %>%   
  dplyr::select(where(is.numeric)) %>%   
  correlate() %>%   
  shave() %>%   
  rplot(print_cor = TRUE) +   
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Export cleaned data set
write_csv(med_clean, "WGU/D209_T2_MV_clean.csv")

#Set seed to reproduce same split
set.seed(123)

#Create indices
train_index <- createDataPartition(med_clean$Initial_days, p=0.8, list=FALSE)

#Create the training and testing data sets
train_set <-med_clean[train_index, ]
test_set <- med_clean[-train_index, ]

#Confirm_split
dim(train_set)
dim(test_set)

#Further split data (y as data frame so I can export to CSV)
x_train <- train_set[, -which(names(train_set) %in% "Initial_days")]
y_train <- as.data.frame(train_set$Initial_days)

x_test <- test_set[, -which(names(test_set) %in% "Initial_days")]
y_test <- as.data.frame(test_set$Initial_days)

#Export as training & testing data
write_csv(x_train, "WGU/D209_T2_MV_xtrain.csv")
write_csv(x_test, "WGU/D209_T2_MV_xtest.csv")
write_csv(y_train, "WGU/D209_T2_MV_ytrain.csv")
write_csv(y_test, "WGU/D209_T2_MV_ytest.csv")

# Convert y back to vector
y_train <- as.vector(unlist(y_train))
y_test <- as.vector(unlist(y_test))

#Train random forest model on training set
rf_model <- randomForest(y_train ~ ., data=x_train, ntree=500)

#Print summary
print(rf_model)

#Make predictions on the testing set
rf_predictions <- predict(rf_model, newdata = x_test)

#Calculate r-squared
ss_res <- sum((y_test - rf_predictions) ^2)
ss_tot <- sum((y_test - mean(y_test)) ^2)
r_squared <- 1 - (ss_res / ss_tot)
print(r_squared)

#RMSE & MSE
rmse <- rmse(y_test, rf_predictions)
print(rmse)

mse <- rmse^2
print(mse)

#Range of Initial days
range_dependent <- range(med_clean$Initial_days)
print(range_dependent)