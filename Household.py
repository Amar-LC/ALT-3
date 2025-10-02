# Importing the package
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Loading the data
df = pd.read_csv('House Price Factors.csv')
df2 = pd.read_csv('House Price Avg.csv')

# Displaying information about the data
print("INFO ON DATA")
df.info()

print()

# Displaying the actual data
print("DATA")
print(df)

# Creating Inputs and Outputs
X = df[['Population', 'Inflation', 'House GDI', 'Migration', 'Crime']] #Inputs
y = df2['Price'] #Outputs

# Splitting the data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=70)

# Creating a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Testing the model by getting it to predict outputs from the X-values in the testing group
y_pred = model.predict(X_test)

# Finding the Root Mean Square Error
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))

# Printing the X and Y values in the testing group
print()
print("X VALUES IN TESTING GROUP")
print(X_test)
print()
print("Y VALUES IN TESTING GROUP")
print(y_test)

# Printing the values the model predicted
print()
print("PREDICTED Y VALUES")
print(y_pred)

# Printing the Root Mean Square Error
print()
print("RMSE")
print(RMSE)

print()
print("JUDGING QUALITY OF RMSE")
Quality = RMSE/(y.max()-y.min())
print(Quality,"here")

while Factor != Population
    Factor = input("""Which factor would you like to see graphed?
    1. Population""")

LineX = np.array(X_train[Factor])
Liney = np.array(y_train)
m, c = np.polyfit(LineX, Liney, 1)
LBF = m * LineX + c #Equation of a straight line: y = mx+c
plt.plot(LineX, LBF)
plt.show()
