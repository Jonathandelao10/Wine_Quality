import numpy as np  # For array manipulation
import pandas as pd  # For easily viewing and manipulating dataframes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import *

wine = pd.read_csv('https://raw.githubusercontent.com/Ashley-Soderlund/Google-Colab-Wine-Quality/main/winequality-red.csv', sep=";")

X = wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
          'pH', 'sulphates', 'alcohol']]
y = wine["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

some_data = X_train.iloc[:5]
some_labels = y_train.iloc[:5]

#print("Predictions:", regr.predict(some_data))

#print(some_labels)

linear_prediction = regr.predict(X_test)

mae = metrics.mean_absolute_error(y_test, linear_prediction)
mse = metrics.mean_squared_error(y_test, linear_prediction)
rmse = np.sqrt(metrics.mean_squared_error(y_test, linear_prediction))

#print("Mean Absolute Error of Linear Regression is ", mae)
#print("Mean Squared Error of Linear Regression is ", mse)
#print("Root Mean Squared Error of Linear Regression is ", rmse)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_prediction = tree.predict(X_test)
#print(classification_report(y_test, tree_prediction))

mae = metrics.mean_absolute_error(y_test, tree_prediction)
mse = metrics.mean_squared_error(y_test, tree_prediction)
rmse = np.sqrt(metrics.mean_squared_error(y_test, tree_prediction))

#print("Mean Absolute Error of Decision Tree Classifier is ", mae)
#print("Mean Absolute Error of Decision Tree Classifier is ", mse)
#print("Mean Absolute Error of Decision Tree Classifier is ", rmse)

forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)

forest_prediction = forest.predict(X_test)
#print(confusion_matrix(y_test, forest_prediction))
#print(classification_report(y_test, forest_prediction))

mae = metrics.mean_absolute_error(y_test, forest_prediction)
mse = metrics.mean_squared_error(y_test, forest_prediction)
rmse = np.sqrt(metrics.mean_squared_error(y_test, forest_prediction))

#print("Mean Absolute Error of Random Forest Classifier is ", mae)
#print("Mean Absolute Error of Random Forest Classifier is ", mse)
#print("Mean Absolute Error of Random Forest Classifier is ", rmse)




#Prediction of user values from front-end function
wine2 = wine.copy()

def prediction(inputwine):
    wine2.append(inputwine)
    X2 = wine2[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']]
    y2 = wine2["quality"]

    #test1 = X2.iloc[-1]
    #print(test1)

    #print(inputwine)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=100)

    forest2 = RandomForestClassifier(n_estimators=100)
    forest2.fit(X_train2, y_train2)
    newresult = forest2.predict(inputwine)
    return newresult[1]