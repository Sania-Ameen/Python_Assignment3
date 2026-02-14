# using the training and testing data from question 3, train a K-Nearest Neighbors classifier
# set the number of neighbors to k = 5
# train the model using the training data then use the trained model to predict the labels of the test data
# compute and display the confusion matrix
# compute and print accuracy
# compute and print precision
# compute and print recall
# compute and print F1-score

# import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# load the data in .csv file into a DataFrame
kidney_disease_dataset_to_dataframe = pd.read_csv("kidney_disease.csv")

# create a feature matrix X containing all columns except classification
x_matrix = kidney_disease_dataset_to_dataframe.drop(columns=["classification"])

# create a label vector y using the classification column
y_label = kidney_disease_dataset_to_dataframe["classification"]

# align the y_labels
y_label = y_label.str.strip()

# turn the categorical features into numerics and handle missing values
x_matrix = pd.get_dummies(x_matrix)
x_matrix = x_matrix.fillna(0)

# split data into training and testing sets
# 70% training and 30% testing
x_train, x_test, y_train, y_test = train_test_split( x_matrix, y_label, test_size = 0.30, random_state = 42)

# set the number of neighbors to 5
number_of_neighbors = 5

# create the knn classifier
knn_classifier = KNeighborsClassifier(n_neighbors = number_of_neighbors) # since k = 5 this will look at the 5 nearest neighbors

# using the training data, train the model then predict the labels of the test data
knn_classifier.fit(x_train,y_train)
predicted_labels_of_the_test_data = knn_classifier.predict(x_test)

# compute confusion matrix, accuracy, precision, recall and F1-score
conf_matrix = confusion_matrix(y_test, predicted_labels_of_the_test_data)
accuracy = accuracy_score(y_test, predicted_labels_of_the_test_data)
precision = precision_score(y_test, predicted_labels_of_the_test_data, pos_label = "ckd")
recall = recall_score(y_test, predicted_labels_of_the_test_data, pos_label = "ckd")
f1 = f1_score(y_test, predicted_labels_of_the_test_data, pos_label = "ckd")

# print the confusion matrix, accuracy, precision, recall and F1-score
print("This is the confusion matrix:\n", conf_matrix)
print("This is the accuracy:", accuracy)
print("This is the precision:", precision)
print("This is the recall:", recall)
print("This is the F1-score:", f1)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# EXPLANATIONS
# In context of kidney disease predictions:
# True positive would mean that the model predicted CKD and the patient has CKD.
# True negative would mean that the model predicted no CKD and the patient has no CKD.
# False positive would mean that the model predicted CKD but the patient has no CKD.
# False negative would mean that the model predicted no CKD but the patient does have CKD.
# Accuracy alone may not be enough to evaluate a classification model because accuracy only measures the general proportion of correct predictions, it doesn't really account for error regions.
# Depending on the data and classes, accuracy alone is not enough to evaluate classification models.
# The most important metric if missing a kidney disease case is recall as it measured the amount of CDK patients are correctly accounted for.
# In terms of real-life scenarios, it is important to identify any false negatives to ensure safe practices.






