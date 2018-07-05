import csv
import random


def loaddataset(filename, split, trainingset = [], testset = []):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])


trainingSet = []
testSet = []
loaddataset('iris.data.csv', 0.66, trainingSet, testSet)
print('Train: ' + repr(len(trainingSet)))
print('Test: ' + repr(len(testSet)))
