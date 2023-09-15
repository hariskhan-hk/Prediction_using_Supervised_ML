# Linear Regression with Python Scikit-Learn

# We will see how the Python Scikit-Learn library for Machine Learning can be used to implement regression functions.

# Simple Linear Regression, as it involves just two variables.

# We will predict the percentage of marks that a student is expected to score based on the number of hours they studied.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading data from a link
data_set = "simple_data_set.csv"
s_data = pd.read_csv(data_set)
print("Data imported successfully")

s_data.head(10)

# Let's plot our data points on a 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data.

# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and the percentage of the score.

# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  

# The next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Finally, Training the Algorithm 
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")

# Plotting the regression line
line = regressor.coef_ * X + regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line)
plt.show()

# Making Predictions

print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 

# The final step is to evaluate the performance of the algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset.
# For simplicity here, we have chosen the Mean Square Error. There are many such metrics.
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))