# -*- coding: utf-8 -*-
"""
Anthony Lasley
CS 450 Machine Learning
Prove 01
"""

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# part 1
iris = datasets.load_iris()

# part 2
dataTrain, dataTest, targetTrain, targetTest = train_test_split(iris.data, iris.target, test_size = 0.3)

class HardCodedClassifier:
    def __init__(self):
        self.type = 0
    def fit(self, data_train, targets_train):
        return "it fits"
    def predict(self, data_test):
        predict=[]
        for x in data_test:
            predict.append(1)
        return predict
            
# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

# part 3

classifier = GaussianNB()
classifier.fit(dataTrain, targetTrain)

# part 4
targetPredicted = classifier.predict(dataTest)
score = accuracy_score(targetPredicted, targetTest)

print("{:.2%}".format(score));


# part 5
fixed_classifier = HardCodedClassifier()
fixed_classifier.fit(dataTrain, targetTrain)
targets_predicted = fixed_classifier.predict(dataTest)

# find accuracy and print
broken_score = accuracy_score(targets_predicted, targetTest)
print("{:.2%}".format(broken_score));

