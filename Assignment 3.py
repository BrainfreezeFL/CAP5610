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
import operator
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
from sklearn import preprocessing
import sklearn.model_selection as model_selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

train_sport = pd.read_csv('Training.csv')
test_sport = pd.read_csv('Testing.csv')
combine_sport = [train_sport, test_sport]
def change_name(input) : 
    temp = []
    #print(input)
    for i in input : 
        if i == 'Texas':
            temp.append('1')
        elif i == 'Virginia':
            temp.append('2')
        elif i == 'GeorgiaTech':
            temp.append('3')
        elif i == 'UMass':
            temp.append('4')
        elif i == 'Clemson':
            temp.append('5')
        elif i == 'Navy':
            temp.append('6')
        elif i == 'USC':
            temp.append('7')
        elif i == 'Temple':
            temp.append('8')
        elif i == 'PITT':
            temp.append('9')
        elif i == 'WakeForest':
            temp.append('10')
        elif i == 'BostonCollege':
            temp.append('11')
        elif i == 'Stanford':
            temp.append('12')
        elif i == 'Nevada':
            temp.append('13')
        elif i == 'MichiganState':
            temp.append('14')
        elif i == 'Duke':
            temp.append('15')
        elif i == 'Syracuse':
            temp.append('16')
        elif i == 'NorthCarolinaState':
            temp.append('17')
        elif i == 'MiamiFlorida':
            temp.append('18')
        elif i == 'Army':
            temp.append('19')
        elif i == 'VirginiaTech': 
            temp.append('20')
        elif i == 'MiamiOhio' :
            temp.append('21')
        elif i == 'NorthCarolina' :
            temp.append('22')
        elif i == 'Georgia' :
            temp.append('23')
        #else :
            #print('Found Something') 
            #print(i)
    #print(temp)
    return temp
def change_place(input) : 
    temp = []
    for i in input : 
        if i == 'Home':
            temp.append('1')
        else : 
            temp.append('2')
    return temp
def change_in_league(input):
    temp = []
    for i in input : 
        if i == 'In':
            temp.append('1')
        else : 
            temp.append('2')
    return temp

def change_media(input):
    temp = []
    for i in input : 
        if i == '1-NBC':
            temp.append('1')
        if i == '4-ABC':
            temp.append('2')
        if i == '3-FOX':
            temp.append('3')
        if i == '2-ESPN':
            temp.append('4')
        if i == '5-CBS':
            temp.append('5')
    return temp
    

def change_label(input) :
    temp = []
    for i in input :
            if i == 'Win':
                temp.append('1')
            else:
                temp.append('2')
    return temp

# Task 1 Question 1
#print(train_sport)

    # Fit The Label Encoder
    # Create a label (category) encoder object
#le = preprocessing.LabelEncoder()

    # Fit the encoder to the pandas column
#le.fit(train_sport['Opponent'])

    # View The Labels
#print(); print(list(le.classes_))

    # Transform Categories Into Integers
    # Apply the fitted encoder to the pandas column
#print(); print(le.transform(train_sport['Opponent']))
train_sport.Opponent = change_name(train_sport['Opponent'])
train_sport.Is_Home_or_Away = change_place(train_sport['Is_Home_or_Away'])
train_sport.Is_Opponent_in_AP25_Preseason = change_in_league(train_sport.Is_Opponent_in_AP25_Preseason)
train_sport.Media = change_media(train_sport.Media)
train_sport.Label = change_label(train_sport.Label)
train_sport = train_sport.drop('Date', axis = 1)
test_sport.Opponent = change_name(test_sport['Opponent'])
test_sport.Is_Home_or_Away = change_place(test_sport['Is_Home_or_Away'])
test_sport.Is_Opponent_in_AP25_Preseason = change_in_league(test_sport.Is_Opponent_in_AP25_Preseason)
test_sport.Media = change_media(test_sport.Media)
test_sport.Label = change_label(test_sport.Label)
test_sport = test_sport.drop('Date', axis = 1)
#print(train_sport)
x_train = train_sport.drop("Label", axis = 1)
y_train = train_sport.Label
x_test = test_sport.drop("Label", axis = 1)
y_test = test_sport.Label
#print(test_sport)
#print(x_test)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


temp = GaussianNB()
prob = temp.fit(x_train, y_train).predict(x_test)
other= temp.fit(x_train, y_train)
print('NB: ')
print(prob)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
print('KNN')
print(pred)


# Task 2 Question 1
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
# uncomment when running the naive bayes
#train_df, test_df = model_selection.train_test_split(train_df, train_size=0.65,test_size=0.35, random_state=101)
#corr_matrix = train_df.corr()
#print(corr_matrix)

print(test_df)

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


test_df = test_df.drop("Name", axis = 1)
test_df = test_df.drop("Ticket", axis = 1)
test_df = test_df.drop("Cabin", axis = 1)
test_df = test_df.drop("PassengerId", axis = 1)

# Change the Sex to a numerical system
test_df['Sex'] = test_df['Sex'].map({'female':1,'male': 0}).astype(int)
test_df['Embarked'].fillna(test_df['Embarked'].dropna().mode()[0], inplace = True)
test_df['Embarked'] = test_df['Embarked'].map({'Q':1,'S': 0, 'C':2}).astype(int)

# Fill in the Age to avoid losing data
test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace = True)

# Fill in the Fare to avoid losing data
test_df['Fare'].fillna(test_df['Fare'].dropna().mean(), inplace = True)
test_df = test_df.astype(int)
y_train = train_df.Survived
x_train = train_df.drop("Survived", axis = 1)
x_test = test_df
print(test_df)
print(x_test)
#features = .columns


temp = GaussianNB()
prob = temp.fit(x_train, y_train)#.predict(x_test)
print('Accuracy : ')
cvs = cross_val_score(prob,x_train,y_train,cv=5, scoring = 'accuracy').mean()
print(cvs)
print('Precision : ')
cvs = cross_val_score(prob,x_train,y_train,cv=5, scoring = 'precision').mean()
print(cvs)
print('Recall : ')
cvs = cross_val_score(prob,x_train,y_train,cv=5, scoring = 'recall').mean()
print(cvs)
print('F1 : ')
cvs = cross_val_score(prob,x_train,y_train,cv=5, scoring = 'f1').mean()
print(cvs)
#other= temp.fit(x_train, y_train)
print('Titanic NB: ')
print(prob)


# Task 2 Question 2
def distance(x, y, columns):
    distance = 0
    for i in range(columns) : 
        distance += pow((x[i] - y[i]),2)
    return math.sqrt(distance)

def getNeighbors(train, testRow, k):
    distanceArray = []
    columns = len(testRow)-1
    x = 0
    train_count = len(train)
    while x != train_count : 
        #print(x)
        #print('Test')
        #print(testRow)
        #print('train')
        #print(train.iloc[x])
        dist = distance(testRow, train.iloc[x], columns)
        distanceArray.append((dist, train.iloc[x]))
        x = x  + 1
    distanceArray.sort(key=operator.itemgetter(0))
    neighbors = []
    x = 0
    while x != k:
        #print(distanceArray[x][1])
        neighbors.append(distanceArray[x][1])
        x = x+1
    return neighbors

def predict(neighbors):
    classdict = {}
    length = len(neighbors)
    x = 0
    while x != length:
        thing = neighbors[x].Survived
        if thing in classdict:
            classdict[thing] += 1
        else : 
            classdict[thing] = 1
        x = x +1
    #print('Classdict')
    #print(classdict)
    sortedVotes = sorted(classdict.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedVotes)
    return sortedVotes[0][0]

def find_accuracy(pred, test) :
    correct = 0
    for x in range(len(test)) :
       # print(pred[x])
        if test.iloc[x].Survived == pred[x] : 
            correct = correct + 1
    return (correct/float(len(test))*100) 

def custom_knn(training, k):
    #train = []
    #test = []
    x_train, x_test = model_selection.train_test_split(training, train_size=0.65,test_size=0.35, random_state=101)
    #print('Train')

    #print(x_train)
    #print('Test')
    #print(x_test)
    predictions = []
    for x in range(len(x_test)):
      #  print(x_test.iloc[x])
        neighbors = getNeighbors(x_train, x_test.iloc[x], k)
     #   print(neighbors)
        pred = predict(neighbors)
        predictions.append(pred)
    
        #print(pred)
    accuracy = find_accuracy(predictions, x_test)
    #(accuracy)
    return k, accuracy
#print(custom_knn(train_df, 3))
k = 1
k_accuracy = []
while k != 100:

    temp = custom_knn(train_df, k)
    k_accuracy.append(temp)
    k = k +1
    print(k_accuracy)
print("The accuracy thing is : ")
print(k_accuracy)