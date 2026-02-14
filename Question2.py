# using the crime.csv dataset and the column ViolentCrimesPerPop create two plots using matplotlib.
# create a histogram showing how values are distributed
# create a boxplot for similar data

# import matplotlib library
# import pandas library
import matplotlib.pyplot as plt
import pandas as pd

crime_dataset_to_dataframe = pd.read_csv("crime1.csv")

# only focus on 'ViolentCrimesPerPop' column
column_ViolentCrimesPerPop = crime_dataset_to_dataframe["ViolentCrimesPerPop"]

# create the histogram
plt.hist(column_ViolentCrimesPerPop, bins=20, color = "pink", edgecolor = "black")
plt.title("Distribution of Values in ViolentCrimesPerPop Dataset")
plt.xlabel("Violent Crimes per Population")
plt.ylabel("Frequency")
# show the histogram
plt.show()

# create the box plot
plt.boxplot(column_ViolentCrimesPerPop)
plt.title("Boxplot of the Distribution of Values in ViolentCrimesPerPop Dataset")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Crimes Conducted per Population")
# show the boxplot
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# EXPLANATIONS
# The histogram shows the data from the 'ViolentCrimesPerPop' column as distributed data.
# The x-axis represents the ranges of the Violent Crimes Per Population from the column data.
# The y-axis represents the frequency of the data in regard to the Violent Crimes Per Population(x_axis)
# The boxplot also shows the distribution of data but in terms of minimum, quartiles, median, and maximum.
# The median is presented with a horizontal line that marks the 50th percentile of the dataset.
# In order words half of the dataset is below this line and the other half is above.
# The boxplot does not suggest any outliers.
