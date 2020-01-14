import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
import operator
from sklearn import preprocessing
import time
'''This is the KNN '''
'''Import Data File'''


Train_Data = np.array(pd.read_csv('./spam_train.csv'))
Test_Data = np.array(pd.read_csv('./spam_test.csv'))
id = Test_Data[:, 0]
#clock = time.clock()
#创建一组特征数据，每一行标识一个样本，每一列标识一个特征
x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]
             ])

#将每一列特征标准化为标准正态分布，注意，标准化是针对每一列而言的
temp1 = preprocessing.scale(Train_Data[:, 0: -1])

temp1 = np.column_stack((temp1, Train_Data[:, -1]))
#print(temp1[0],temp1.shape)
Train_Data=temp1
#Train_Data = np.insert(temp, Train_Data.shape[1], Train_Data[:, -1], axis=1)
temp2 = preprocessing.scale(Test_Data[:, 1: -1])
#print(temp2.shape)
temp2= np.column_stack((temp2, Test_Data[:, -1]))
Test_Data= temp2
#Test_Data = np.insert(temp, -1, Test_Data[:, -1], axis=1)

#print(Train_Data.shape)
#print(Test_Data.shape)

#print(Train_Data[0],'123',Test_Data[0])

#x=input()
#print(x_scale.mean(axis=1))
#x=input()
# 可以查看标准化后的数据的均值与方差，已经变成0,1了
#print(temp.mean(axis=0))
# axis=1表示对每一行去做这个操作，axis=0表示对每一列做相同的这个操作
#print(x_scale.mean(axis=1))
# 同理，看一下标准差
#print(temp.std(axis=0))
# 调用fit方法，根据已有的训练数据创建一个标准化的转换器
scaler = preprocessing.StandardScaler().fit(x)



# Note that test_data has extra attribute in index of 0
def Cal_Dist(Matrix_A, Matrix_B):
    dist = np.sqrt(np.sum(np.square(Matrix_A - Matrix_B)))
    return dist


def Get_Nearest_Node(Train, Test, k):
    Index_dist = []
    dist_test = []
    for test in Test:  # traver all the elements in Test
        for train in Train:  # traver all the elements in Train
            dist = Cal_Dist(test[0:-1], train[0:-1])
            Index_dist.append([train[-1], dist])  # recode distance and index of train
        dist_test.append(Index_dist)
        Index_dist = []
    a = [[1.0, 752.1809486499907], [1.0, 2.893000518492867], [1.0, 1287.9678916859689]]
    a.sort(key=operator.itemgetter(1))
    #print(a[0:2])
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
    #print(len(Test_Data))
    #print(NN[1404])  # Only need Classification
    return NN


def Vote(Predict, k):
    Predicted = []    # initialize array to store predicted results
    for i in range(0, len(Predict)):     # for each of the
        Vote = 0
        for j in range(0, k):
            if Predict[i][j] == 1.0:
                Vote += 1
            else:
                Predict[i][j] == 0.0
                Vote -= 1
        if Vote >= 1:
            #Predicted.append(1.0)
            Predicted.append('spam')
        else:
            #Predicted.append(0.0)
            Predicted.append('no')


    return Predicted


def Cal_Accuracies(Test, Predict):
    if len(Test) != len(Predict):
        print("Error")
    accuracy = 0.0
    for i in range(len(Test)):
        if Test[i][-1] == Predict[i]:
            accuracy += 1
    #print(accuracy)
    accuracy = accuracy / len(Test)
    return accuracy


def main():
    Pred_R= []  # Results of Prediction within all cases of k
    for k in (1, 5, 11, 21, 41, 61, 81, 101, 201, 401):
        Nearest_Node = Get_Nearest_Node(Train_Data, Test_Data, k)
        Predict = Vote(Nearest_Node, k)  # for each k, get the prediction
        Pred_R.append(Predict)  # Each prediction is an array, storing the array to the
    Pred_R = np.array(Pred_R)
    print(id.shape)
    print(Pred_R.T.shape)
    R = np.column_stack((id, Pred_R.T))
    print(R[0:50])


main()

# x=[[[1,2,3],45],[[2,3,4],44]]
# print(x[0][0])
