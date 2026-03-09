# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset Import required libraries and read the Salary.csv dataset.

2.Prepare the Data Separate the independent variable (Position Level) as X and the dependent variable (Salary) as y.

3.Train the Model Create a Decision Tree Regressor model and fit it with X and y.

4.Predict the Salary Use the trained model to predict the salary for a given position level.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RAKSHATHA S A
RegisterNumber: 212225220079
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv("C:/Users/acer/Downloads/Salary.csv")

X = data.iloc[:, 1:2].values   
y = data.iloc[:, 2].values    


regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

predicted_salary = regressor.predict([[6.5]])
print("Predicted Salary:", predicted_salary)


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Decision Tree Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
```

## Output:
<img width="765" height="624" alt="image" src="https://github.com/user-attachments/assets/87fafb22-29d4-4ac8-b06a-8e57aae8dc08" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
