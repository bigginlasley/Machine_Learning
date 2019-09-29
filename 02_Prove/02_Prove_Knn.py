from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import operator
import math
# find the euclidean distance between the arrays
def getDistance(x, y, length):
    distance = 0
    for i in range(length):
        distance += ((x[i] - y[i]) ** 2)
    return math.sqrt(distance)

# grab the nearest neighbor based on k
def getNeighbors(dataTrain, testValue, k):
    distances = []
    length = len(testValue)-1

    for x in range(len(dataTrain)):
        distance = getDistance(testValue, dataTrain[x], length)
        distances.append((dataTrain[x], distance))
    distances.sort(key=operator.itemgetter(1))
    neigbors = []
    # once sorted, loops through and adds on the distance and value
    # to the neighbors
    for i in range(k):
        neigbors.append(distances[i][0])
    return neigbors


def getAnswer(neighbor):
    checker = {}
    for i in range(len(neighbor)):
        answer = neighbor[i][-1]
        #if answer is already in the dictionary increase it by 1
        if answer in checker:
            checker[answer]+=1
        #else if the answer isn't in the dictionary add it and set it to 1
        else:
            checker[answer]=1
    val = sorted(checker.items(), key=operator.itemgetter(1), reverse=True)
    return val[0][0]

# variable declarations
low = 0
high = 5
testSet = []
trainSet = []
predict = []

# load and split the data
iris = datasets.load_iris()
dataTrain, dataTest, targetTrain, targetTest = train_test_split(iris.data, iris.target, test_size = .30)


# combine the data and target into one array
for i in range(len(dataTest)):
    testSet.append(np.append(dataTest[i], targetTest[i]))

for i in range(len(dataTrain)):
    trainSet.append(np.append(dataTrain[i], targetTrain[i]))

# Promtp user for K value
while True:
    try:
        print("")
        number = int(input("Please enter K value: "))
        if low < number <= high:
            break
        else:
            print("Value is not in range. Range is between 0 - 6")
    except ValueError:
            print("Invalid Input")

# loop through data finding the nearest neighbor for each 
# then predicting which classification it is
for i in range(len(dataTest)):
    neigbors = getNeighbors(trainSet, testSet[i], number)
    answer = int(getAnswer(neigbors))
    predict.append(answer)

# built in classifier
classifier = GaussianNB()
classifier.fit(dataTrain, targetTrain)
targetPredicted = classifier.predict(dataTest)

#find accuracy of both
score = accuracy_score(targetPredicted, targetTest)
myScore = accuracy_score(predict, targetTest)

# format and print out the accuracy
print("Built in: " +"{:.2%}".format(score) + " vs: " + "{:.2%}".format(myScore))



