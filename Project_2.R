#Loading the dataset
data("ToothGrowth")

#load the libraries 
library(caret)
library(class)
library(e1071)
library(rpart)
library(neuralnet)
library(ggplot2)
View(ToothGrowth)
#Convert species to a numeric format for neural network
ToothGrowth$supp <- as.factor(ToothGrowth$supp)    

#Set seed for reproducibility
set.seed(123)

#split the data into training (70%) and testing (30%) sets
trainIndex <- createDataPartition(ToothGrowth$supp, p = 0.7, list = FALSE)     
trainData <- ToothGrowth[trainIndex,] 
testData <- ToothGrowth[-trainIndex,]

#1)Applying KNN algorithm: -
  #Implementing knn 
  knn_pred <- knn(train = trainData[, c("dose", "len")],
                  test = testData[, c("dose", "len")],
                  cl = trainData$supp, k = 3)
  #Creating Confusion Matrix :-
  confusion_matrix=table(knn_pred,testData$supp)
  # Finding the accuracy
  knn_accuracy=sum(diag(confusion_matrix))/sum(confusion_matrix)
  # Finding the precision
  knn_precision <- diag(confusion_matrix) / colSums(confusion_matrix)
  #Finding the recall
  knn_recall <- diag(confusion_matrix) / rowSums(confusion_matrix)
  #Finding the f1 score
  knn_f1 <- 2 * (knn_precision * knn_recall) / (knn_precision + knn_recall) 
  #Finding the error_rate
  knn_error =( 1 - knn_accuracy)
 
  
  
# 2)Applying Naïve Bayes algorithm: -

  
#Train the model
nb_model <- naiveBayes(supp ~ dose + len, data = trainData)
#prepare the test set for prediction
nb_pred <- predict(nb_model, testData)
#create a confusion matrix 
confusion_matrix=table(nb_pred,testData$supp)

# Finding the accuracy
nb_accuracy=sum(diag(confusion_matrix))/sum(confusion_matrix)
# Finding the precision
nb_precision <- diag(confusion_matrix) / colSums(confusion_matrix)
#Finding the recall
nb_recall <- diag(confusion_matrix) / rowSums(confusion_matrix)
#Finding the f1 score
nb_f1 <- 2 * (nb_precision * nb_recall) / (nb_precision + nb_recall) 
#Finding the error_rate
nb_error <- 1 - nb_accuracy



#3)Applying Decision Tree algorithm: -

#Train the model
tree_model <- rpart(supp ~ dose + len, data = trainData, method = "class")
#Prepare the test set for prediction
tree_pred <- predict(tree_model, testData, type = "class")
#Creating Confusion Matrix
confusion_matrix=table(tree_pred,testData$supp)
# Finding the accuracy
tree_accuracy=sum(diag(confusion_matrix))/sum(confusion_matrix)
# Finding the precision
tree_precision <- diag(confusion_matrix) / colSums(confusion_matrix)
#Finding the recall
tree_recall <- diag(confusion_matrix) / rowSums(confusion_matrix)
#Finding the f1 score
tree_f1 <- 2 * (tree_precision *tree_recall) / (tree_precision +tree_recall) 
#Finding the error_rate
tree_error <- 1 - tree_accuracy




#4)Applying Neural Network algorithm: -
  # Normalize 'len' and 'dose' to make them suitable for neural networks

  trainData$len <- scale(trainData$len)
  testData$len <- scale(testData$len)
  trainData$dose <- scale(trainData$dose)
  testData$dose <- scale(testData$dose)
  
  #Train the model
  nn_model <- neuralnet(supp ~ dose + len, data = trainData, hidden =c(3))
  #Prepare the train set for prediction
  nn_pred <- predict(nn_model, testData, type = "class")
  test_pre=predict(nn_model,newdata=testData)
  #Convert prediction to class labels
  predicted_classes=apply(test_pre,1,which.max)
  #Create Confusion Matrix
  confusion_matrix=table(Predicted=predicted_classes,Actual=testData$supp)
  # Finding the accuracy
  nn_accuracy=sum(diag(confusion_matrix))/sum(confusion_matrix)
  # Finding the precision
  nn_precision <- diag(confusion_matrix) / colSums(confusion_matrix)
  #Finding the recall
  nn_recall <- diag(confusion_matrix) / rowSums(confusion_matrix)
  #Finding the f1 score
  nn_f1 <- 2 * (nn_precision * nn_recall) / (nn_precision + nn_recall) 
  #Finding the error_rate
  nn_error <- 1 - nn_accuracy
  plot(nn_model)
  
  
  
  


#VISUALIZATION

# Create the boxplot

ggplot(ToothGrowth, aes(x = dose, y = len, fill = supp)) +
  geom_boxplot() +
  labs(title = "Boxplot of Tooth Length by Dose and Supplement Type",
       x = "Dose (mg)",
       y = "Tooth Length",
       fill = "Supplement") +
  theme_minimal()

# Histogram for tooth length
ggplot(ToothGrowth, aes(x = len)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Tooth Length",
       x = "Tooth Length",
       y = "Frequency") +
  theme_minimal()

# Density plot for tooth length
ggplot(ToothGrowth, aes(x = len, fill = supp)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Tooth Length by Supplement Type",
       x = "Tooth Length",
       y = "Density",
       fill = "Supplement") +
  theme_minimal()


# Comparative Analysis
Model = c("K-Nearest Neighbor", "Naive Bayes", "Decision Tree", "Neural Network")
Accuracy = c(knn_accuracy, nb_accuracy, tree_accuracy, nn_accuracy)
Precision=c(knn_precision, nb_precision, tree_precision, nn_precision)
Recall=c(knn_recall,nb_recall,tree_recall,nn_recall)
F1_score=c(knn_f1,nb_f1,tree_f1,nn_f1)
Error=c(knn_error,nb_error,tree_error,nn_error)


Model
Accuracy
Precision  
F1_score  
Recall  
Error

#COMPARISON
comparison_df <- data.frame(
  Model,
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1_score", "Error"), times = 4),
  Value = c(knn_accuracy, nb_accuracy, tree_accuracy, nn_accuracy,
            mean(knn_precision, na.rm = TRUE), mean(nb_precision, na.rm = TRUE), mean(tree_precision, na.rm = TRUE), mean(nn_precision, na.rm = TRUE),
            mean(knn_recall, na.rm = TRUE), mean(nb_recall, na.rm = TRUE), mean(tree_recall, na.rm = TRUE), mean(nn_recall, na.rm = TRUE),
            mean(knn_f1, na.rm = TRUE), mean(nb_f1, na.rm = TRUE), mean(tree_f1, na.rm = TRUE), mean(nn_f1, na.rm = TRUE),
            knn_error, nb_error, tree_error, nn_error)
)

ggplot(comparison_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position ="dodge", color = "black") +
  theme_minimal() +
  labs(title = "Model Comparison using Algorithms",
       x = "Model",
       y = "Metric Value") +
  scale_fill_brewer(palette = "Set3") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

  
  
  