Description of Files 

# Question 1: 
This program takes a .csv file (crime1.csv) and loads it into a DataFrame. It then looks at one column from the dataset and computes, then prints the mean, median, maximum, minimum and the standard deviation. 

# Question 2: 
This program imports the libraries pandas and matplotlib.pyplot, takes a .csv file (crim1.csv) and the same column.
It then creates a histogram, as well as a boxplot. Both plots depict the data found in the 'ViolentCrimesPerPop' column. 

# Question 3:
This porgram loads a .csv file (kidney_disease.csv) into a dataframe, creates a feauture matrix x that includes all columns except for the column labelled 'classification', then created a label vecotor y and splits the data into training and testing sets. 
Verifying output is then printed. 

# Question 4: 
This program follows up program 3 by using the training and testing data. 
Using kNeighbousClassifier, it then trains the model using the training data, and then uses the model to predict labels of the tested data. 
It then computes and prints the confusion matrix, accuracy, percision, recall and F1 score. 

# Question 5: 
This last program uses essential code from programs 3 and 4 to train multiple KNN models then compute as well as test accuracy. 
Using a for loop, the program loops through the given k-values, prints a small table with each values corresponding accuracy and then calculates the k-value which gives the highest accuracy. 
