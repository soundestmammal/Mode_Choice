# Logistic Regression

from collections import Counter
Counter(y_pred)
Counter(y_test)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data_V51.csv')
dataset = dataset.drop(['Unnamed: 0'], axis=1)
dataset = dataset[dataset.INCOM_R != 9]
dataset = dataset.fillna("")
dataset = dataset[dataset.LIC != 'DK']
dataset = dataset.drop(['LIC'])
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
le = LabelEncoder()

labelencoder_x1 = LabelEncoder()
X[:, 5] = labelencoder_x1.fit_transform(X[:, 5])
labelencoder_x2 = LabelEncoder()
X[:, 6] = labelencoder_x2.fit_transform(X[:, 6])
labelencoder_x3 = LabelEncoder()
X[:, 7] = labelencoder_x3.fit_transform(X[:, 7])

dataset = dataset.dropna()
X = X.fillna("")
dataset = dataset.fillna("")
X[5]
X = np.asarray(X)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, multi_class='ovr')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
