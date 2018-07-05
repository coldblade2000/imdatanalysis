import csv
import random
import math
import operator


def loaddataset(filename, split, trainingset=[], testset=[]):
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
loaddataset('iris_data.csv', 0.66, trainingSet, testSet)
print('Train: ' + repr(len(trainingSet)))
print('Test: ' + repr(len(testSet)))


def euclideandistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)


data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideandistance(data1, data2, 3)
print('Distance: ' + repr(distance))


def getneighbors(trainingset, testinstance, k):
    distances = []
    length = len(testinstance)-1
    for x in range(len(trainingSet)):
        dist = euclideandistance(testinstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

