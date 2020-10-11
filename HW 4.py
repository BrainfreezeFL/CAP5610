# Samuel Lewis
# HW 2
# Machine Learning Class
# I dowloaded the anaconda pack to run this

# Read in all of the data
import pandas as pd
import math
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn import tree
import pydotplus
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#%matplotlib inline
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

corr_matrix = train_df.corr()
#print(corr_matrix)



#print(train_df.describe())
# Drop the unnecessary features
train_df = train_df.drop("Name", axis = 1)
train_df = train_df.drop("Ticket", axis = 1)
train_df = train_df.drop("Cabin", axis = 1)
train_df = train_df.drop("PassengerId", axis = 1)

# Change the Sex to a numerical system
train_df['Sex'] = train_df['Sex'].map({'female':1,'male': 0}).astype(int)
train_df['Embarked'].fillna(train_df['Embarked'].dropna().mode()[0], inplace = True)
train_df['Embarked'] = train_df['Embarked'].map({'Q':1,'S': 0, 'C':2}).astype(int)

# Fill in the Age to avoid losing data
train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace = True)

# Fill in the Fare to avoid losing data
train_df['Fare'].fillna(train_df['Fare'].dropna().mean(), inplace = True)
train_df = train_df.astype(int)
test = train_df.Survived
other = train_df.drop("Survived", axis = 1)
features = other.columns

y_train = train_df.Survived
x_train = train_df.drop("Survived", axis = 1)
x_test = test_df

linear_svc = svm.SVC(kernel='linear')
temp = linear_svc.fit(x_train, y_train)
cvs = cross_val_score(temp,x_train,y_train,cv=5, scoring = 'accuracy').mean()
print(cvs)
print(linear_svc.kernel)

linear_svc = svm.SVC(kernel='poly', degree = 2)
temp = linear_svc.fit(x_train, y_train)
cvs = cross_val_score(temp,x_train,y_train,cv=5, scoring = 'accuracy').mean()
print(cvs)
print(linear_svc.kernel)

linear_svc = svm.SVC(kernel='rbf')
temp = linear_svc.fit(x_train, y_train)
cvs = cross_val_score(temp,x_train,y_train,cv=5, scoring = 'accuracy').mean()
print(cvs)
print(linear_svc.kernel)