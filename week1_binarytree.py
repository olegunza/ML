import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

data_used = data[['Pclass', 'Fare', 'Sex', 'Age', 'Survived']].dropna(axis=0)
data_Y = data_used['Survived']
data_X = data_used[['Pclass', 'Fare', 'Sex', 'Age']] 
data_X = data_X.replace({'male': 1, 'female': 2})
clf = DecisionTreeClassifier(random_state=241)
clf.fit(data_X, data_Y)

importances = clf.feature_importances_

print(importances)