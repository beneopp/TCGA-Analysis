import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def ExtractConversionFromJson(loadedjson):
    dictlist = []
    for sample in loadedjson:
        singledict = {}
        fileid = sample['file_id']
        caseid = sample['associated_entities'][0]['case_id']
        singledict[fileid] = caseid
        dictlist.append(singledict)
    return dictlist

def CreateOrderedClasses(clin, dictlist, fileids):
    classes = []
    for _, row in fileids.iterrows():
        fileid = row[0]
        for dict in dictlist:
            if fileid in dict:
                caseid = dict[fileid]
        for _, row2 in clin.iterrows():
            matchid = row2[0]
            if caseid == matchid:
                status = row2[15]
                if status == 'Dead':
                    classes.append(1)
                elif status == 'Alive':
                    classes.append(0)
                else:
                    raise Exception('Something went wrong')
    return classes

def MinMaxNormalization(data):
    min = np.amin(data, axis = 0)
    max = np.amax(data, axis = 0)
    for i in range(len(data)):
        data[i] = (data[i] - min) / (max - min)
    return data

def GradientDescent(data, classes, step_size, iterations):
    coeffs = np.zeros(len(data[0])+1)  #+1 accounts for intercept at index 0
    for iteration in range(iterations):
        for i in range(len(data)):
            condprob = LogisticPredict(data[i], coeffs)
            coeffs[0] = coeffs[0] + step_size * (classes[i] - condprob) * condprob * (1 - condprob)  #intercept gradient descent - no data multiplication
            for j in range(len(data[i])):
                coeffs[j + 1] = coeffs[j + 1] + step_size * (classes[i] - condprob) * condprob * (1 - condprob)  * data[i][j]  #loss function gradient descent
    return coeffs

def LogisticPredict(sample, coeffs):
    b = coeffs[0]
    for i in range(len(coeffs)-1):
        b += coeffs[i + 1] * sample[i]
    sigmoid = 1 / (1 + np.exp(-b))
    return sigmoid

def LogisticRegression(traindata, trainclasses, testdata, step_size, iterations):
    predictedclasses = []
    coeffs = GradientDescent(traindata, trainclasses, step_size, iterations)
    for sample in testdata:
        print(LogisticPredict(sample, coeffs))
        prediction = round(LogisticPredict(sample, coeffs)) # < 0.5 = 0, > 0.5 = 1
        predictedclasses.append(prediction)
    return predictedclasses

def TestAccuracy(classes, predictedclasses):
    correct = 0
    for i in range(len(classes)):
        if classes[i] == predictedclasses[i]:
            correct += 1
    accuracy = correct / len(classes)
    return accuracy

def CrossValidation(matrix, classes):
    matrix = np.column_stack((matrix, classes))
    np.random.shuffle(matrix)
    classes = matrix[:,-1]
    matrix = np.delete(matrix, -1, 1)
    accuracylist = []

    testdata = np.split(matrix, [35])[0]
    traindata = np.split(matrix, [35])[1]
    predictedclasses = LogisticRegression(traindata, classes[35:], testdata, 0.1, 20)
    accuracy = TestAccuracy(classes[:35], predictedclasses)
    print("%", accuracy)
    accuracylist.append(accuracy)

    testdata = np.split(matrix, [35, 70])[1]
    traindata = np.concatenate((np.split(matrix, [35, 70])[0], np.split(matrix, [35, 70])[2]))
    predictedclasses = LogisticRegression(traindata, np.concatenate((classes[:35], classes[70:])), testdata, 0.1, 20)
    accuracy = TestAccuracy(classes[35:70], predictedclasses)
    print("%",accuracy)
    accuracylist.append(accuracy)

    testdata = np.split(matrix, [70, 105])[1]
    traindata = np.concatenate((np.split(matrix, [70, 105])[0], np.split(matrix, [70, 105])[2]))
    predictedclasses = LogisticRegression(traindata, np.concatenate((classes[:70], classes[105:])), testdata, 0.1, 20)
    accuracy = TestAccuracy(classes[70:105], predictedclasses)
    print("%",accuracy)
    accuracylist.append(accuracy)

    testdata = np.split(matrix, [105, 141])[1]
    traindata = np.concatenate((np.split(matrix, [105, 141])[0], np.split(matrix, [105, 141])[2]))
    predictedclasses = LogisticRegression(traindata, np.concatenate((classes[:105], classes[141:])), testdata, 0.1, 20)
    accuracy = TestAccuracy(classes[105:141], predictedclasses)
    print("%",accuracy)
    accuracylist.append(accuracy)

    testdata = np.split(matrix, [141])[1]
    traindata = np.split(matrix, [141])[0]
    predictedclasses = LogisticRegression(traindata, classes[:141], testdata, 0.1, 20)
    accuracy = TestAccuracy(classes[141:], predictedclasses)
    print("%",accuracy)
    accuracylist.append(accuracy)

    return accuracylist

def FeatureSelection(matrix, classes):
    coeffs = GradientDescent(np.split(matrix, [35])[1], classes[35:], 0.1, 20)
    maxindices = (-coeffs).argsort()[:25]
    print((-coeffs).argsort()[:3])
    minindices = coeffs.argsort()[:25]
    print(coeffs.argsort()[:3])
    indices = np.concatenate((maxindices, minindices))
    newmatrix = matrix[:, indices]
    return newmatrix

with open("metadata.cart.2021-05-01.json", "r") as read_file:
    loadedjson = json.load(read_file)

matrix = np.loadtxt('matrix.txt')
matrix = MinMaxNormalization(matrix)
clinicaloutcome = pd.read_csv('clinical.tsv', sep='\t')
clinicaloutcome.drop_duplicates(subset=['case_id'], inplace=True)
fileids = pd.read_csv('sample_names.txt', header=None)

classes = CreateOrderedClasses(clinicaloutcome, ExtractConversionFromJson(loadedjson), fileids)
trimmedmatrix = FeatureSelection(matrix, classes)
untrimmed = CrossValidation(matrix, classes)
trimmed = CrossValidation(trimmedmatrix, classes)
averageuntrimmed = round(np.average(untrimmed), 2)
averagetrimmed = round(np.average(trimmed), 2)

Xlabels = ['1','2','3','4','5']
Y = untrimmed
Z = trimmed
X = np.arange(len(Xlabels))

plt.bar(X - 0.2, Y, 0.4)
plt.bar(X + 0.2, Z, 0.4)
plt.xticks(X, Xlabels)
plt.xlabel('Cross-Validation Trial')
plt.ylabel('Prediction Accuracy')
colors = {'55125 Features':'blue', '50 Features':'orange'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.title(f"Average Accuracy - 55125 Features: {averageuntrimmed}%, Average Accuracy - 50 Features: {averagetrimmed}%", fontsize=10)
plt.savefig('Accuracy.pdf')
