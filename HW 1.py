# Samuel Lewis
# HW 1
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
#%matplotlib inline
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
#print(combine)

# Gather what question we want the results for
print("What question are you answering : ")
question = input()

# Using this for debugging
if question == '100' : 
    for x in train_df:
        print(x)

# Question 2
if question == '2':
    print(train_df.Ticket)

# Question 5
elif question == '5' :
    # We want a value to see if each column header has blank info
    blank = False
    stack = []
    stack_one = []

    # We want to loop through all of the column headers
    for x in train_df:

        # We want to loop through each of the rows in each column to see if any are empty
        for y in train_df[x] : 
            
            if y == "":
                blank = True
             
            elif y == None:
                blank = True
               
            elif isinstance(y,str) == False:
                if math.isnan(y) == True:
                    blank = True
                  
        if blank == True:
            stack.append(x)
        blank = False
    print("Training data with Blanks : ")
    print(stack)
    stack = []
    for x in test_df:

        # We want to loop through each of the rows in each column to see if any are empty
        for y in test_df[x] : 
            
            if y == "":
                blank = True
              
            elif y == None:
                blank = True
             
            elif isinstance(y,str) == False:
                if math.isnan(y) == True:
                    blank = True
                
        if blank == True:
            stack.append(x)
        blank = False
    print("Testing data with Blanks : ")
    print(stack)

elif question == '6' :
    stack = []

    # We want to loop through all of the column headers
    for x in train_df:

        # We want to loop through each of the rows in each column to see if any are empty
        stack.append(x + " : " + str(type(train_df[x][1])))
    print(stack)

elif question == '7' : 
    
    # We want to loop through all of the column headers
    #for x in train_df:

    #print("Age" + " : " + str(numpy.std(train_df.Age)))
    #age = train_df.groupby('Age')
    print(train_df.Age.describe())
    print(train_df.SibSp.describe())
    print(train_df.Parch.describe())
    print(train_df.Fare.describe())

elif question == '8' : 
    test = train_df
    #test.describe(include=[object])
    print(test.astype(str).describe(include=object))
    #print(train_df.Survived.describe())
    #print(train_df.Pclass.describe())
    #print(train_df.Name.describe())
    ##print(train_df.Sex.describe())
    #print(train_df.Ticket.describe())
    #print(train_df.Embarked.describe())
    #print(train_df.Cabin.describe())

elif question == '9' : 

    #new = train_df.Pclass == 1
    new = train_df[train_df.Pclass==1]
    print(new.astype(str).describe(include=object))
    # Then divide the amoutn that survived by the total people to get the correlation
    #newer = new.loc[:,'Pclass', 'Survived']
    #print(new)
    #print(train_df.corr())

elif question == '10' : 
    print("Test Results Are : ")
    new = test_df[test_df.Sex=='female']
    print(new.astype(str).describe(include=object))
    print("Training results are : ")
    new = train_df[train_df.Sex=='female']
    print(new.astype(str).describe(include=object))
    # Then divide the number of women who died by the number of women

elif question == '11' :
    
    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    
    plt.show()

elif question == '12' :
   
    graph = sns.FacetGrid(train_df, col='Survived', row='Pclass')
    graph.map(plt.hist, 'Age', bins=20)
    #x = train_df.filter(regex='Age|Survived|Pclass')
    #x.hist(by=(x.Survived and x.Pclass))
    #plt.title("Survived")
    ##plt.xlabel("Age")
    #plt.ylabel("Count")
    plt.show()

elif question == '13' : 

    graph = sns.FacetGrid(train_df, row = 'Survived', col='Embarked')
    graph.map(sns.barplot, 'Sex', 'Fare')
    #graph.addLegend()
    plt.show()

elif question == '14' : 

    graph = sns.FacetGrid(train_df, col ='Survived')
    graph.map(plt.hist, 'Ticket')
    plt.show()

elif question == '15' : 
    count = 0
    for y in test_df.Cabin :
        #print(y)

        if isinstance(y,str) == False:
            if math.isnan(y) == True:
                count = count + 1
    for y in train_df.Cabin :
       # print(y)

        if isinstance(y,str) == False:
            if math.isnan(y) == True:
                count = count + 1

    print("The amount of NaN in Cabin is : " + str(count))

    print(train_df.describe(include = ['O']))
    print(test_df.describe(include = ['O']))

elif question == '16' : 

    for x in combine : 
        x['Sex'] = x['Sex'].map({'female':1,'male': 0}).astype(int)

    print(combine)

elif question == '17' : 
    train_df = train_df.drop("Name", axis = 1)
    train_df = train_df.drop("Ticket", axis = 1)
    train_df = train_df.drop("Embarked", axis = 1)
    train_df = train_df.drop("Cabin", axis = 1)

    train_df['Sex'] = train_df['Sex'].map({'female':1,'male': 0}).astype(int)
    for x in train_df : 

        train_df[x] = pd.to_numeric(train_df[x])
    print(train_df.Age)
    imputer = KNNImputer(n_neighbors = 3)
    train_df = imputer.fit_transform(train_df)
    for x in train_df : 
        print(x[4])
    print("This output is the age data")


elif question == '18' : 
    #print(train_df.Embarked)
    # We know that the most frequently used port is S based on previous answers, so I will hard code all of the NaN to be S
    for x in train_df.Embarked : 
        if isinstance(x,str) == False:
            if math.isnan(x) == True or x == "":
                x = 'S'
                # There are two values that need to be replaced so this checks out.
                print("Was replaced")
   # print(train_df.Embarked)

elif question == '19' : 

    z = test_df[test_df.Name == 'Storey, Mr. Thomas']
    print(z)
    test_df['Fare'].fillna(test_df['Fare'].dropna().mode()[0], inplace = True)

    z = test_df[test_df.Name == 'Storey, Mr. Thomas']
    print(z)

elif question == '20' : 

    test_df['Fare'].fillna(test_df['Fare'].dropna().mode()[0], inplace = True)
    combine = [train_df, test_df]
    for x in combine : 
        x.loc[ x['Fare'] <= 7.91, 'Fare'] = 0
        x.loc[ (x['Fare'] <= 14.454) & (x['Fare'] > 7.91), 'Fare'] = 1
        x.loc[ (x['Fare'] <= 31.0) & (x['Fare'] > 14.454), 'Fare'] = 2
        x.loc[ (x['Fare'] <=512.329) &  (x['Fare'] > 31.0), 'Fare'] = 3
        x['Fare'] = x['Fare'].astype(int)

    print(combine)

else :
    print("No code was needed for that question")