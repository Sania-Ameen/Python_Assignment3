# load crime.csv into a pandas DataFrame and focus on the column ViolentCrimesPerPop
# Compute and print the following:
# mean, median, standard deviation, minimum value, maximum value

# import pandas library
import pandas as pd

crime_dataset_to_dataframe = pd.read_csv("crime1.csv")

# only focus on 'ViolentCrimesPerPop' column
column_ViolentCrimesPerPop = crime_dataset_to_dataframe["ViolentCrimesPerPop"]

# compute and print calculations using the columns data
print("The mean is:", column_ViolentCrimesPerPop.mean())
print("The meadian is:", column_ViolentCrimesPerPop.median())
print("The minimum value is:", column_ViolentCrimesPerPop.min())
print("The maximum value is:", column_ViolentCrimesPerPop.max())
print("The standard deviation is:", column_ViolentCrimesPerPop.std())

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# EXPLANATIONS
# The distribution between the mean and median seems to be right-skewed because the mean is slightly bigger than the median.
# This indicates that the mean is generally pulled higher due to the bigger values in the given dataset.
# The mean is more affected than the median, because the mean is the average of all the given data and minimum or maximum values will cause it to skew right or left.
