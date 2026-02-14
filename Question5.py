# using the same training and testing data, train multiple KNN models using different values of k
# k = 1, 3, 5, 7, 9
# for each value of k, compute the test accuracy and store the results
# create a small table showing each value of k and its corresponding accuracy
# identify which value of k gives the highest test accuracy

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

# do multiple k-values (1,3,5,7,9)
# make an empty list to store the new accuracy results
list_of_k_values = [1,3,5,7,9]
accuracy_results = []

for k_values in list_of_k_values:
    # compute the test accuracy for each value
    knn_classifier = KNeighborsClassifier(n_neighbors = k_values)
    knn_classifier.fit(x_train,y_train)
    predicted_labels_of_the_test_data = knn_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predicted_labels_of_the_test_data)

    # append all new accuracy in the empty list
    accuracy_results.append(accuracy)

# show the new table with each k value and its corresponding accuracy
# create a DataFrame and print
k_values_with_corresponding_accuracies_dataframe= pd.DataFrame({
    "k_values": list_of_k_values ,
    "Accuracy": accuracy_results
})
print(k_values_with_corresponding_accuracies_dataframe)

# identify which value of k gives the highest test accuracy
highest_k_value = list_of_k_values[accuracy_results.index(max(accuracy_results))]
print("The values of k which gives the highest test accuracy is:", highest_k_value)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# EXPLANATIONS
# Changing the k-value affects the behaviour of the model by causing it to look at either fewer or more neighbours, thus resulting in the corresponding data class.
# Smaller k-values may cause overfitting because the model is not used to larger scaled data.
# This would cause the model to be inaccurate with newer data.
# Larger k-values may cause underfitting because the model considers values that nearly the size of the dataset, thus restricting the model to analyze more discrete patterns.