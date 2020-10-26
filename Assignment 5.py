import math
import random
import time
from tkinter import *


# Read in all of the data
import pandas as pd
import math
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import euclidean_distances

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
iris = pd.read_csv('iris.csv')
combine = [train_df, test_df]

corr_matrix = train_df.corr()




######################################################################
# This section contains functions for loading CSV (comma separated values)
# files and convert them to a dataset of instances.
# Each instance is a tuple of attributes. The entire dataset is a list
# of tuples.
######################################################################

# Loads a CSV files into a list of tuples.
# Ignores the first row of the file (header).
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
#   fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0] # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset

# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    # separate the fields
    lineList = cleanLine.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple

# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])

# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
  if len(s) == 0:
    return False
  if  len(s) > 1 and s[0] == "-":
      s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
######################################################################
# This section contains functions for clustering a dataset
# using the k-means algorithm.
######################################################################
# 1 for euclidean
# 2 for manhatten distance
# 3 for cosine similarity
def distance(instance1, instance2, dis_type):
    if dis_type == 1:
        if instance1 == None or instance2 == None:
            return float("inf")
        sumOfSquares = 0
        for i in range(1, len(instance1)):
            sumOfSquares += (instance1[i] - instance2[i])**2
    elif dis_type == 2:
        #print(instance1)
        #print(instance2)
        if instance1 == None or instance2 == None:
            return float("inf")
        sumOfSquares = 0
        #for i in range(1, len(instance1)):
        instance1 = list(instance1)
        instance2 = list(instance2)
        a = instance1.pop(0)
        a = instance2.pop(0)
        sumOfSquares += manhattan_distances([instance1], [instance2])
    elif dis_type == 3:
        #print(instance1)
        #print(instance2)
        if instance1 == None or instance2 == None:
            return float("inf")
        sumOfSquares = 0
        #for i in range(1, len(instance1)):
        instance1 = list(instance1)
        instance2 = list(instance2)
        a = instance1.pop(0)
        a = instance2.pop(0)
        sumOfSquares += (1-cosine_similarity([instance1], [instance2]))
    elif dis_type == 4:
        #print(instance1)
        #print(instance2)
        if instance1 == None or instance2 == None:
            return float("inf")
        sumOfSquares = 0
        #for i in range(1, len(instance1)):
        instance1 = list(instance1)
        instance2 = list(instance2)
        a = instance1.pop(0)
        a = instance2.pop(0)
        sumOfSquares += (1-jaccard(instance1, instance2))
    #print("The centroid is :")
    #print(instance2)
    #print("The SSE is :")
    #sse = 0
    #for i in range(1, len(instance1)):
    #    sse += (instance1[i] - instance2[i])**2
    #print(sse)
    return sumOfSquares



def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)

def assign(instance, centroids):
    minDistance = distance(instance, centroids[0], 1)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i], 1)
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAll(instances, centroids):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids)
        clusters[clusterIndex].append(instance)
    return clusters

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids
def accuracy(clusters):
    count = 0
    right = 0
    wrong = 0
    one = 0
    two = 0
    three = 0
    accuracy = 0
    for i in clusters:
        for j in range(0,len(i)):
            #count = count + 1
            if i[j][4] == 0: 
                one = one + 1
                count = count + 1
            if i[j][4] == 1 :
                two = two +1
                count = count + 1
            if i[j][4] == 2 : 
                three = three + 1
                count = count + 1
        if one < two :
            temp = "two"
            if two < three :
                temp = "three"
        elif one < three :
            temp = "three"
        else : 
            temp = "one"
        print("This cluster is :")
        print(temp)
        print("Count is : ")
        print(count)
        print("One is :")
        print(one)
        print("Two is : ")
        print(two)
        print("Three is :")
        print(three)
        if count > 0:
            print(max(one, two, three))
            print(max(one,two,three)/count)
            
            accuracy = accuracy + (max(one,two,three)/count)*(count/151)

        count = 0
        one = 0
        two = 0
        three = 0   
    return accuracy
        

def kmeans(instances, k, animation=False, initCentroids=None):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
        #print(centroids)
    else:
        centroids = initCentroids
    prevCentroids = []
    prevsse = 1000000
    currentsse = 1000000
    if animation:
        delay = 1.0 # seconds
        canvas = prepareWindow(instances)
        clusters = createEmptyListOfLists(k)
        clusters[0] = instances
        paintClusters2D(canvas, clusters, centroids, "Initial centroids")
        time.sleep(delay)
    iteration = 0
    while (centroids != prevCentroids and iteration < 100):#(currentsse >= prevsse):#centroids != prevCentroids):
        #print("Centroids")
        #print(centroids)
        #print("PrevCentroids")
        #print(prevCentroids)
       # print("THE ITERATION IS : ")
       # print(iteration)
        iteration += 1
        clusters = assignAll(instances, centroids)
        #print('clusters')
        #print(clusters)
        if animation:
            paintClusters2D(canvas, clusters, centroids, "Assign %d" % iteration)
            time.sleep(delay)
        prevCentroids = centroids
        
        centroids = computeCentroids(clusters)
        for i in centroids:
      #      print(i)
            if type(i) == float:
                i = round(list(i),2)
     #   print("Iteration : ")
     #   print(iteration)
     #   print("Centroids")
     #   print(centroids)
        withinss = computeWithinss(clusters, centroids)
        prevsse = currentsse
        currentsse = withinss

        if animation:
            paintClusters2D(canvas, clusters, centroids,
                            "Update %d, withinss %.1f" % (iteration, withinss))
            time.sleep(delay)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["iterations"] = iteration
    print("Iteration")
    temp = accuracy(clusters)
    print("The final accuracy is : ")
    print(temp)
    print(iteration)

    return result

def computeWithinss(clusters, centroids):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += distance(centroid, instance, 1)
    #print("Centroid SSE: ")
    #print(result)
    return result

# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n+1):
        print("k-means trial %d," % i , end = "")
        trialClustering = kmeans(instances, k)
        print("withinss: %.1f" % trialClustering["withinss"])
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    print("Trial with minimum withinss:", minWithinssTrial)
    return bestClustering


######################################################################
# This section contains functions for visualizing datasets and
# clustered datasets.
######################################################################

def printTable(instances):
    for instance in instances:
        if instance != None:
            line = instance[0] + "\t"
            for i in range(1, len(instance)):
                line += "%.2f " % instance[i]
            print(line)

def extractAttribute(instances, index):
    result = []
    for instance in instances:
        result.append(instance[index])
    return result

def paintCircle(canvas, xc, yc, r, color):
    canvas.create_oval(xc-r, yc-r, xc+r, yc+r, outline=color)

def paintSquare(canvas, xc, yc, r, color):
    canvas.create_rectangle(xc-r, yc-r, xc+r, yc+r, fill=color)

def drawPoints(canvas, instances, color, shape):
    random.seed(0)
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2*margin) / (maxX - minX)
    scaleY = float(height - 2*margin) / (maxY - minY)
    for instance in instances:
        x = 5*(random.random()-0.5)+margin+(instance[1]-minX)*scaleX
        y = 5*(random.random()-0.5)+height-margin-(instance[2]-minY)*scaleY
        if (shape == "square"):
            paintSquare(canvas, x, y, 5, color)
        else:
            paintCircle(canvas, x, y, 5, color)
    canvas.update()

def connectPoints(canvas, instances1, instances2, color):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2*margin) / (maxX - minX)
    scaleY = float(height - 2*margin) / (maxY - minY)
    for p1 in instances1:
        for p2 in instances2:
            x1 = margin + (p1[1]-minX)*scaleX
            y1 = height - margin - (p1[2]-minY)*scaleY
            x2 = margin + (p2[1]-minX)*scaleX
            y2 = height - margin - (p2[2]-minY)*scaleY
            canvas.create_line(x1, y1, x2, y2, fill=color)
    canvas.update()

def mergeClusters(clusters):
    result = []
    for cluster in clusters:
        result.extend(cluster)
    return result

def prepareWindow(instances):
    width = 500
    height = 500
    margin = 50
    root = Tk()
    canvas = Canvas(root, width=width, height=height, background="white")
    canvas.pack()
    canvas.data = {}
    canvas.data["margin"] = margin
    setBounds2D(canvas, instances)
    paintAxes(canvas)
    canvas.update()
    return canvas

def setBounds2D(canvas, instances):
    attributeX = extractAttribute(instances, 1)
    attributeY = extractAttribute(instances, 2)
    canvas.data["minX"] = min(attributeX)
    canvas.data["minY"] = min(attributeY)
    canvas.data["maxX"] = max(attributeX)
    canvas.data["maxY"] = max(attributeY)

def paintAxes(canvas):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    canvas.create_line(margin/2, height-margin/2, width-5, height-margin/2,
                       width=2, arrow=LAST)
    canvas.create_text(margin, height-margin/4,
                       text=str(minX), font="Sans 11")
    canvas.create_text(width-margin, height-margin/4,
                       text=str(maxX), font="Sans 11")
    canvas.create_line(margin/2, height-margin/2, margin/2, 5,
                       width=2, arrow=LAST)
    canvas.create_text(margin/4, height-margin,
                       text=str(minY), font="Sans 11", anchor=W)
    canvas.create_text(margin/4, margin,
                       text=str(maxY), font="Sans 11", anchor=W)
    canvas.update()


def showDataset2D(instances):
    canvas = prepareWindow(instances)
    paintDataset2D(canvas, instances)

def paintDataset2D(canvas, instances):
    canvas.delete(ALL)
    paintAxes(canvas)
    drawPoints(canvas, instances, "blue", "circle")
    canvas.update()

def showClusters2D(clusteringDictionary):
    clusters = clusteringDictionary["clusters"]
    centroids = clusteringDictionary["centroids"]
    withinss = clusteringDictionary["withinss"]
    canvas = prepareWindow(mergeClusters(clusters))
    paintClusters2D(canvas, clusters, centroids,
                    "Withinss: %.1f" % withinss)

def paintClusters2D(canvas, clusters, centroids, title=""):
    canvas.delete(ALL)
    paintAxes(canvas)
    colors = ["blue", "red", "green", "brown", "purple", "orange"]
    for clusterIndex in range(len(clusters)):
        color = colors[clusterIndex%len(colors)]
        instances = clusters[clusterIndex]
        centroid = centroids[clusterIndex]
        drawPoints(canvas, instances, color, "circle")
        if (centroid != None):
            drawPoints(canvas, [centroid], color, "square")
        connectPoints(canvas, [centroid], instances, color)
    width = canvas.winfo_reqwidth()
    canvas.create_text(width/2, 20, text=title, font="Sans 14")
    canvas.update()


######################################################################
# Test code
######################################################################
print("1")
dataset = loadCSV("iris.csv")
#print(dataset)
dataset1 = list(map(tuple, dataset))
#print(type(dataset1))
for i in range(0, len(dataset1)):
    dataset1[i] = list(dataset1[i])
    #print(dataset1[i])
    #print(dataset1[i][4])
    if dataset1[i][4] == 'virginica':
        dataset1[i][4] = 0
    elif dataset1[i][4] == 'versicolor':
        dataset1[i][4] = 1
    elif dataset1[i][4] == 'setosa':
        dataset1[i][4] = 2


#print(dataset1)
print("2")
#showDataset2D(dataset)
print("3")
# Change the values in the distance functions located throughout the program based on what distance function you want to use
# 1 = euclidean
# 2 = Manhatten
# 3 = Cosine
# 4 = Jacard
# 5 = 
# Task 1 Part 1
clustering = kmeans(dataset1, 5, True)
# Task 1 Part 3
#clustering = kmeans(dataset, 2, True, [('Centroid1',3,3),('Centroid2', 8,3)])
# Task 1 Part 4
#clustering = kmeans(dataset, 2, True, [('Centroid1',3,2),('Centroid2', 4,8)])
# Task 1 Part 2
#clustering = kmeans(dataset, 2, True, [('Centroid1',4,6),('Centroid2', 5,4)])
# Task 2 Part 1
#clustering = repeatedKMeans(dataset1,5,100)
print(clustering)

print("4")
#printTable(clustering["centroids"])
print("5")
def maximum(one, two):
    max_dist = 0
    for op in one:
        for tp in two:
            x1 = op[0]
            y1 = op[1]
            x2 = tp[0]
            y2 = tp[1]
           # print("The distance between these points:")
            #print(x1,y1)
            #print(x2,y2)

            temp = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            #print(temp)
            if temp > max_dist:
                max_dist = temp
    return max_dist
def minimum(one, two):
    min_dist = 100000000000000
    for op in one:
        for tp in two:
            x1 = op[0]
            y1 = op[1]
            x2 = tp[0]
            y2 = tp[1]
           # print("The distance between these points:")
           # print(x1,y1)
           # print(x2,y2)

            temp = math.sqrt((x2-x1)**2 + (y2-y1)**2)
           # print(temp)
            if temp < min_dist:
                min_dist = temp
    return min_dist

def average(one, two):
    avg_dist = 0
    iter = 0
    for op in one:
        for tp in two:
            x1 = op[0]
            y1 = op[1]
            x2 = tp[0]
            y2 = tp[1]
           # print("The distance between these points:")
           # print(x1,y1)
           # print(x2,y2)

            temp = math.sqrt((x2-x1)**2 + (y2-y1)**2)
           # print(temp)
            iter = iter+1
            avg_dist = avg_dist + temp
    return avg_dist/iter

red = ((4.7, 3.2), (4.9,3.1), (5.0, 3.0), (4.6,2.9))
blue = ((5.9,3.2), (6.7,3.1), (6.0,3.0), (6.2,2.8))

#print(maximum(red,blue))
#print(minimum(red,blue))
#print(average(red,blue))