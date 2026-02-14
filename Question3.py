# load kidney.diseases.csv into a pandas DataFrame and create a feature matrix x that contains all columns except classification
# create a label vector y using the classification column
# splot the dataset into training data (70%) and testing data (30%)


# import libraries panda and train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split

# load data in .csv into a DataFrame
kidney_disease_dataset_to_dataframe = pd.read_csv("kidney_disease.csv")

# create a feature matrix X containing all columns except classification
x_matrix = kidney_disease_dataset_to_dataframe.drop(columns=["classification"])

# create a label vector y using the classification column
y_label = kidney_disease_dataset_to_dataframe["classification"]

# split data into training and testing sets
# 70% training and 30% testing
x_train, x_test, y_train, y_test = train_test_split( x_matrix, y_label, test_size = 0.3, random_state = 50)

# print to verify rows and columns match (shape)
print("x_matrix_train shape:", x_train.shape)
print("x_matrix_test shape:", x_test.shape)
print("y_label_train shape:", y_train.shape)
print("y_label_test shape:", y_test.shape)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# EXPLANATIONS
# Training and testing should not be done on the same data because the model would already know all the examples with their outcomes when trained so when it is tested it is already aware of the answers.
# This would mean that the model didn't actually imply what was trained because it memorized the outcomes rather than actually understanding the patterns.
# The purpose of the testing set is to ensure that the model actually understands the adequate patterns to further use on new datasets.
