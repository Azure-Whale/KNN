import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
import operator
from scipy import stats

'''This is the KNN '''
'''Import Data File'''

Train_Data = np.array(pd.read_csv('./spam_train.csv'))  # Import Train Data
Test_Data = np.array(pd.read_csv('./spam_test.csv'))  # Import Test Data


# Note that test_data has extra attribute in index of 0
def Cal_Dist(Matrix_A, Matrix_B):  # Method of Calculating Euclidean distance
    dist = np.sqrt(np.sum(np.square(Matrix_A - Matrix_B)))
    return dist


def Get_Nearest_Node(Train, Test, k):
    Classification_dist = []
    dist_test = []
    for test in Test:  # traver all the elements in Test
        for train in Train:  # traver all the elements in Train
            dist = Cal_Dist(test
                            [1:-1], train[0:-1])
            Classification_dist.append([train[-1], dist])  # recode distance and index of train
        dist_test.append(Classification_dist)  # Now, for each test instance
        # we give an array including all distances to train nodes and their labels to it
        Classification_dist = []   # Reset array for next use of storage
    # a = [[1.0, 752.1809486499907], [1.0, 2.893000518492867], [1.0, 1287.9678916859689]]
    # a.sort(key=operator.itemgetter(1))
    for i in range(0, len(dist_test)):
        dist_test[i].sort(key=operator.itemgetter(
            1))  # Note that you are going to sort for each instance instead of the all instance.
    # dist_test.sort(key=operator.itemgetter(1))

    NN = []
    temp = []
    for j in range(0, len(dist_test)):  # Traver all
        for i in range(0, k):  # select first k element (sorted)
            temp.append(dist_test[j][i][0])  # select the classification
        NN.append(temp)
        temp = []
    # print(len(Test_Data))
    # print(NN[1404])  # Only need Classification
    return NN


def Vote(Predict, k):
    Predicted = []  # initialize array to store predicted results
    for i in range(0, len(Predict)):  # for each of the
        Vote = 0
        for j in range(0, k):
            if Predict[i][j] == 1.0:
                Vote += 1
            else:
                Predict[i][j] == 0.0
                Vote -= 1
        if Vote >= 1:
            Predicted.append(1.0)
        else:
            Predicted.append(0.0)
    # print(Predicted)
    return Predicted


def Cal_Accuracies(Test, Predict):
    if len(Test) != len(Predict):
        print("Error")
    accuracy = 0.0
    for i in range(len(Test)):
        if Test[i][-1] == Predict[i]:
            accuracy += 1
    # print(accuracy)
    accuracy = accuracy / len(Test)
    return accuracy


def main():
    for k in (1, 5, 11, 21, 41, 61, 81, 101, 201, 401):
        Nearest_Node = Get_Nearest_Node(Train_Data, Test_Data, k)
        Predict = Vote(Nearest_Node, k)
        Accuracy = Cal_Accuracies(Test_Data, Predict)
        print('The Accuracy for k = ', k, ' is ', Accuracy, ' %\n')


main()
