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

# Displaying data
print("FACTORS DATA")
print(df)
print()
print("PRICE DATA")
print(df2)

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

Factor = 0 #Initialising variable

#Asking user to select which factor to graph against house prices
# While loop allows the question to be repeated in the case of incorrect user inputs
while Factor != 'Population' or Factor != 'Inflation' or Factor != 'House gdi' or Factor != 'Migration' or Factor != 'Crime':
    Factor = input("""Which factor would you like to see graphed?
    1. Population
    2. Inflation
    3. House GDI
    4. Migration
    5. Crime
    """).capitalize() #.capitalize() takes into consideration the different capitalisation methods of different users when typing
    if Factor == 'Population' or Factor == 'Inflation' or Factor == 'House gdi' or Factor == 'Migration' or Factor == 'Crime':
        break 
    print()
    print("That is not a factor available. Please ensure spelling is correct.")
    print()
   
# Change the value of factor if user chooses House GDI as the .capitalize() earlier will turn the G, D, and I lowercase
if Factor == 'House gdi': 
    Factor = 'House GDI'

# Creating a line of best fit
LineX = np.array(X_train[Factor])
Liney = np.array(y_train)
m, c = np.polyfit(LineX, Liney, 1)
LBF = m * LineX + c #Equation of a straight line: y = mx+c
plt.scatter(LineX, Liney, color='red')
# Giving labels for the x and y axis
if Factor == 'Population':
    plt.xlabel(Factor + " per millions")
if Factor == 'Inflation':
    plt.xlabel(Factor + " in %")
if Factor == 'House GDI':
    plt.xlabel(Factor)
if Factor == 'Migration':
    plt.xlabel(Factor + " per thousands")
if Factor == 'Crime':
    plt.xlabel(Factor)
plt.ylabel("House Prices per thousand")
plt.plot(LineX, LBF)
plt.show()
