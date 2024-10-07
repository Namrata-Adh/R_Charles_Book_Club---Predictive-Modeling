
###### Charles Book Club ######



#__________________________________________________________________________________________________#

###### PHASE 1: Data Import ######
#__________________________________________________________________________________________________#


# Step 1: Load the data set using Text(readr). ################
#CharlesBookClub.xlsx#



# Step 2: Rename and make a new copy of the data set (For Simplicity).  ##############
CBC <- CharlesBookClub_2_
View(CBC)




# Step 3: Load the packages to be used. ##################
library(reshape2)
library(viridis)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(finalfit)
library(scales)
library(caret)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(FNN)
library(fastDummies)
library(FactoMineR)
library(factoextra)



#__________________________________________________________________________________________________#

###### PHASE 2 : Exploratory Data Analysis. ##########
#__________________________________________________________________________________________________#


# Step 1: Explore the overall data set #########################
glimpse(CBC)
# This gives the row/column names/number and data types.



# Step 2: Identify Missing values #########################

missing_glimpse(CBC)  # To identify missing values in each variables.
missing_plot(CBC)    # To visualize missing values in the data set.

# There are no missing values #



# Step 3: Descriptive Statistics (To check data distribution and outliers) ###################
summary(CBC)



#__________________________________________________________________________________________________#

######  Box-Plots and Histograms #######
#__________________________________________________________________________________________________#


# Let's Visualize distributions using box plots and histograms

###  Melt the data into long-format for easier plotting ############
CBC_melt <- melt(CBC, id.vars = c("Seq#", "ID#"))

# Seq# and ID# are identifier variables.



### Create box plots for each variable ###############
ggplot(CBC_melt, aes(x = "", y = value, fill = variable)) + 
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free") + 
  labs(title = "Boxplots for All Variables", x = "Variables", y = "Values") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", margin = margin(b = 20)),
    strip.text = element_text(size = 8, face = "bold"),
    legend.position = "none")



# Histogram
ggplot(CBC_melt, aes(x = value, fill = variable)) + 
  geom_histogram(color = "black", alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") + 
  labs(title = "Histograms for All Variables", x = "Values", y = "Frequency") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", margin = margin(b = 20)),
    strip.text = element_text(size = 8, face = "bold"),
    legend.position = "none"
  )


#__________________________________________________________________________________________________#

###### PHASE 3 : Data Cleaning/Preparation ##########

#__________________________________________________________________________________________________#


# Remove unnecessary variables ##################
CBC <- CBC %>% select(-`Seq#`, -`ID#`) 



# Define the normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))}



# Normalize the numerical columns
numerical_columns <- sapply(CBC, is.numeric)
CBC[numerical_columns] <- lapply(CBC[numerical_columns], normalize)

summary(CBC)




# For DIMENSION REDUCTION #


# Correlation Matrix #############
cor_table <- round(cor(CBC),2)
View(cor_table)    



# Melt the correlation matrix #############
melted_cor_matrix <- melt(cor_table)



# Create the heat-map ####################
ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradientn(colors = c("red", "orange", "white", "cornflowerblue", "beige"),
                       values = scales::rescale(c(-1, -0.5, 0, 0.5, 1)),
                       name = "Correlation") +
  labs(title = "Correlation Heatmap of CharlesBookClub Data", x = "Variables", y = "Variables") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5, margin = margin(b = 10)),
    axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1)
  ) +
  coord_fixed() +
  geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 2.5)



# Perform PCA
pca_result <- PCA(CBC[, sapply(CBC, is.numeric)], scale.unit = TRUE, graph = FALSE)



# Summary of PCA
summary(pca_result)



# Plot PCA
fviz_pca_ind(pca_result, geom.ind = "point", pointshape = 21, 
             
             pointsize = 2, fill.ind = "blue", col.ind = "black", 
             
             palette = "jco", addEllipses = TRUE, label = "var", 
             
             col.var = "black", repel = TRUE)



fviz_pca_var(pca_result, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             
             repel = TRUE)




#__________________________________________________________________________________________________#

###### Data Modeling : Data preparation #####
#__________________________________________________________________________________________________#


# Step 1: Change the target variable into factor data type ############
CBC$Florence<- factor(CBC$Florence, levels = c(1, 0), 
                      labels = c("Book Purchaser", "Non-Purchaser"))




# Step 2: Partition the data into training (70%) and test sets (30%) #################
set.seed(123)  # For reproducibility
train.index <- sample(1:nrow(CBC), 0.7 * nrow(CBC))
CBC.train<- CBC[train.index, ]
CBC.valid <- CBC[-train.index, ]



#__________________________________________________________________________________________________#

###### Data Modeling : Logistic Regression Model #####

###### MODEL 1: With all variables ######
#__________________________________________________________________________________________________#



# Run logistic regression ##################################
logistic_model <- glm(`Florence` ~ ., data = CBC.train, family = "binomial") 
options(scipen=999)

summary(logistic_model)




# Predict on validation and training set #####################
logit.pred.valid <- predict(logistic_model, CBC.valid, type = "response") # Validation
logit.pred.train <- predict(logistic_model, CBC.train, type = "response") # Training




# Convert probabilities to predicted classes with cutoff = 0.5 #################
cutoff <- 0.5
train.pred.class <- ifelse(logit.pred.train >= cutoff, "Book Purchaser", "Non-Purchaser")
valid.pred.class <- ifelse(logit.pred.valid >= cutoff, "Book Purchaser", "Non-Purchaser")



# Ensure the levels match ####################
train.pred.class <- factor(train.pred.class, levels = c( "Book Purchaser", "Non-Purchaser"))
valid.pred.class <- factor(valid.pred.class, levels = c("Book Purchaser", "Non-Purchaser"))



# Generate confusion matrix for training set #######################
train.confusion.matrix <- confusionMatrix(train.pred.class, 
                                          CBC.train$Florence, positive = "Book Purchaser")
print(train.confusion.matrix)




# Generate confusion matrix for validation set #########################
valid.confusion.matrix <- confusionMatrix(valid.pred.class, 
                                          CBC.valid$Florence, positive = "Book Purchaser")
print(valid.confusion.matrix)




# First 5 actual and predicted records for validation set ##################
predicted_classes <- ifelse(logit.pred.valid >= 0.5, "Book Purchaser", "Non-Purchaser")
predicted_classes <- factor(predicted_classes, levels = c("Non-Purchaser", "Book Purchaser"))



# Display the first 5 actual and predicted records
results <- data.frame(actual = CBC.valid$Florence[1:20], predicted = predicted_classes[1:20])
print(results)



#__________________________________________________________________________________________________#

###### Data Modeling : Logistic Regression Model #####

###### MODEL 2: Only significant variables ######
#__________________________________________________________________________________________________#


# Run logistic regression ##################################
logistic_model_refined <- glm(Florence ~ Gender + F + ChildBks + CookBks + ArtBks + GeogBks,
                              data = CBC.train, family = "binomial") 
options(scipen=999)

summary(logistic_model_refined)




# Predict on validation and training set #####################
logit.pred.valid.refined <- predict(logistic_model_refined, CBC.valid, type = "response") # Validation
logit.pred.train.refined <- predict(logistic_model_refined, CBC.train, type = "response") # Training




# Convert probabilities to predicted classes with cutoff = 0.5 #################
cutoff <- 0.5
train.pred.class.refined <- ifelse(logit.pred.train.refined >= cutoff, "Book Purchaser", "Non-Purchaser")
valid.pred.class.refined <- ifelse(logit.pred.valid.refined >= cutoff, "Book Purchaser", "Non-Purchaser")




# Ensure the levels match ####################
train.pred.class.refined <- factor(train.pred.class.refined, levels = c("Book Purchaser", "Non-Purchaser"))
valid.pred.class.refined <- factor(valid.pred.class.refined, levels = c("Book Purchaser", "Non-Purchaser"))



# Generate confusion matrix for training set #######################
train.confusion.matrix.refined <- confusionMatrix(train.pred.class.refined, CBC.train$Florence, positive = "Book Purchaser")
print(train.confusion.matrix.refined)



# Generate confusion matrix for validation set #########################
valid.confusion.matrix.refined <- confusionMatrix(valid.pred.class.refined, CBC.valid$Florence, positive = "Book Purchaser")
print(valid.confusion.matrix.refined)



# First 5 actual and predicted records for validation set ##################
predicted_classes_refined <- ifelse(logit.pred.valid.refined >= 0.5, "Book Purchaser", "Non-Purchaser")
predicted_classes_refined <- factor(predicted_classes_refined, levels = c("Non-Purchaser", "Book Purchaser"))


results_refined <- data.frame(actual = CBC.valid$Florence[1:5], predicted = predicted_classes_refined[1:5])
print(results_refined)




#__________________________________________________________________________________________________#

###### Data Modeling : KNN Neighbor #####

######  Only significant variables ######
#__________________________________________________________________________________________________#


# Ensure Florence is a factor for classification
# Ensure Florence is a factor with labels "non-purchaser" and "book purchaser"
CBC.train$Florence <- factor(CBC.train$Florence, levels = c(1, 0), labels = c( "book purchaser", "non-purchaser"))
CBC.valid$Florence <- factor(CBC.valid$Florence, levels = c(1, 0), labels = c("book purchaser", "non-purchaser"))




# Find the best value of K
set.seed(123)
k_fold <- trainControl(method = "cv", number = 10)

k_seq <- seq(1, 20, by = 1)

knn_model <- train(Florence ~ ., data = CBC.train, method = "knn", 
                   trControl = k_fold, tuneGrid = data.frame(k = k_seq))

knn_model



# Train the KNN model with k = 4
set.seed(123)
knn_model <- train(Florence ~ ., data = CBC.train, method = "knn", 
                   trControl = trainControl(method = "none"), 
                   tuneGrid = data.frame(k = 4))



# Make predictions on the training set
knn_predictions_train <- predict(knn_model, CBC.train)



# Make predictions on the validation set
knn_predictions_valid <- predict(knn_model, CBC.valid)



# Generate the confusion matrix for the training set
confusion_matrix_train <- confusionMatrix(knn_predictions_train, CBC.train$Florence, positive = "book purchaser")
print(confusion_matrix_train)



# Generate the confusion matrix for the validation set
confusion_matrix_valid <- confusionMatrix(knn_predictions_valid, CBC.valid$Florence, positive = "book purchaser")
print(confusion_matrix_valid)


#__________________________________________________________________________________________________#

###### Data Modeling : Neural Nets #####

######  Only significant variables ######
#__________________________________________________________________________________________________#


# Step 1: Normalize the data  ##########################
summary(CBC)



# Step 2: Partition the data into training (70%) and validation (30%) sets #####################
set.seed(123)  # set seed for reproducibility
train.index <- sample(1:nrow(CBC), 0.7 * nrow(CBC))  
NN.CBC.train <- CBC[train.index, ]
NN.CBC.valid <- CBC[-train.index, ]



# Step 3: Train the neural network #################
NN_Model <- neuralnet(Florence ~ Gender + M + R + F + FirstPurch + ChildBks + DoItYBks +
                        RefBks + ArtBks + GeogBks + ItalCook + ItalAtlas + ItalArt + RelatedPurchase, 
                      data = CBC.train, hidden = 2)


# Plot the model
plot(NN_Model)



# Make predictions on the training set ####################
train.predictions <- compute(NN_Model, CBC.train[, -which(names(CBC.train) == "Florence")])
train.predicted.classes <- ifelse(train.predictions$net.result > 0.5,"Book Purchases", "Non-Purchaser")



# Convert the target variable into factor 
CBC.train$Florence <- factor(CBC.train$Florence, levels = c(1, 0), 
                             labels = c("Book Purchases", "Non-Purchaser"))
train.predicted.classes <- factor(train.predicted.classes, levels = c(1, 0), 
                                  labels = c("Book Purchases", "Non-Purchaser"))



# Confusion matrix for training set ################################
confusion_matrix_train <- confusionMatrix(train.predicted.classes, CBC.train$Florence)
print(confusion_matrix_train)



# Make predictions on the validation set #########################
valid.predictions <- compute(NN_Model, NN.CBC.valid[, -which(names(NN.CBC.valid) == "Florence")])
valid.predicted.classes <- ifelse(valid.predictions$net.result > 0.5, 1, 0)



# Convert the target variable into factor 
NN.CBC.valid$Florence <- factor(NN.CBC.valid$Florence, levels = c(1, 0), 
                                labels = c("Book Purchases", "Non-Purchaser"))
valid.predicted.classes <- factor(valid.predicted.classes, levels = c(1, 0), 
                                  labels = c("Book Purchases", "Non-Purchaser"))




# Confusion matrix for validation set ###########################
confusion_matrix_valid <- confusionMatrix(valid.predicted.classes, NN.CBC.valid$Florence)
print(confusion_matrix_valid)

