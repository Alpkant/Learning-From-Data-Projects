import numpy as np
import matplotlib.pyplot as plt
from numpy import *



csvfile = np.genfromtxt('datatrain.csv', delimiter = ',')
traindata = csvfile[1:,][:] #Read only data part of the csv

csvfile = np.genfromtxt('datatrain.csv', delimiter = ',')
testdata = csvfile[1:,][:] #Read test set of the csv

class0data = []
class1data = []
class2data = []


classCounter = np.zeros(3,dtype=integer)
meanVector = np.zeros((3,2))
Z = np.empty((0,),dtype=int)


#I calculate mean vectors and split the class datas to different lists
for i in traindata:
    if i[2] == 0:
        meanVector[0][0]+=i[0]
        meanVector[0][1]+=i[1]
        class0data = np.vstack([class0data,i]) if len(class0data) else i
        classCounter[0]+=1
    if i[2] == 1:
        meanVector[1][0]+=i[0]
        meanVector[1][1]+=i[1]
        class1data = np.vstack([class1data,i]) if len(class1data) else i
        classCounter[1]+=1
    if i[2] == 2:
        meanVector[2][0]+=i[0]
        meanVector[2][1]+=i[1]
        class2data = np.vstack([class2data,i]) if len(class2data) else i
        classCounter[2]+=1


for i in range(3):
    meanVector[i]=np.divide(meanVector[i],classCounter[i]) #We got mean vector for all classes

n =np.shape(traindata)[0]

class0covarianceMatrix = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        for k in range(classCounter[0]):
            class0covarianceMatrix[i][j]+=(class0data[k][i]-meanVector[0][i])*(class0data[k][j]-meanVector[0][j])
        class0covarianceMatrix[i][j] = class0covarianceMatrix[i][j]/(classCounter[0]-1)


class1covarianceMatrix = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        for k in range(classCounter[1]):
            class1covarianceMatrix[i][j]+=(class1data[k][i]-meanVector[1][i])*(class1data[k][j]-meanVector[1][j])
        class1covarianceMatrix[i][j] = class1covarianceMatrix[i][j]/(classCounter[1]-1)

class2covarianceMatrix = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        for k in range(classCounter[2]):
            class2covarianceMatrix[i][j]+=(class2data[k][i]-meanVector[2][i])*(class2data[k][j]-meanVector[2][j])
        class2covarianceMatrix[i][j] = class2covarianceMatrix[i][j]/(classCounter[2]-1)

#Before calculating cost function i calculate class probabilities
class0Probability = classCounter[0]/ classCounter.sum(axis=0)
class1Probability = classCounter[1]/ classCounter.sum(axis=0)
class2Probability = classCounter[2]/ classCounter.sum(axis=0)

def classPrediction(x):
    global Z
    x = np.transpose(x)

    #g(x) cost functions for each class according to the function that i mentioned in report
    g_x0_ci  =  np.dot(np.dot((-0.5)*np.transpose(x-meanVector[0]),np.linalg.inv(class0covarianceMatrix)), (x-meanVector[0]) )- 0.5*np.log(np.absolute(np.linalg.det(class0covarianceMatrix))) +np.log(class0Probability)
    g_x1_ci  =  np.dot(np.dot((-0.5)*np.transpose(x-meanVector[1]),np.linalg.inv(class1covarianceMatrix)), (x-meanVector[1]) )- 0.5*np.log(np.absolute(np.linalg.det(class1covarianceMatrix))) +np.log(class1Probability)
    g_x2_ci  =  np.dot(np.dot((-0.5)*np.transpose(x-meanVector[2]),np.linalg.inv(class2covarianceMatrix)), (x-meanVector[2]) )- 0.5*np.log(np.absolute(np.linalg.det(class2covarianceMatrix))) +np.log(class2Probability)
    # print("0=",g_x0_ci)
    # print("1=",g_x1_ci)
    # print("2=",g_x2_ci)

    if   g_x0_ci > g_x1_ci and g_x0_ci > g_x2_ci:
        predict = 0
    elif g_x1_ci > g_x0_ci and g_x1_ci > g_x2_ci:
        predict = 1
    else:
        predict = 2

    Z = np.append(Z,predict)


def plot_the_result():
    global Z
    h = .05
    x_min, x_max = traindata[:, 0].min() ,  traindata[:, 0].max()
    y_min, y_max = traindata[:, 1].min() ,  traindata[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max,h), np.arange(y_min, y_max,h))
    
    for i in range(np.shape(xx)[0]):
        for j in range(np.shape(xx)[1]):
            classPrediction([xx[i][j],yy[i][j]])

    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks()
    plt.yticks()
    plt.draw()

def main():
    global Z
    numberofTrue=0
    numberofFalse=0

    plot_the_result()
    Z = np.empty((0,),dtype=int)
    for i in range(np.shape(testdata)[0]):
        classPrediction([testdata[i][0],testdata[i][1]])
        if testdata[i][2] == Z[i]:
            numberofTrue+=1
        else:
            numberofFalse+=1

    print("Accuracy in the test dataset is : ",100*numberofTrue/(numberofTrue+numberofFalse))
    plt.show()
if __name__ == "__main__":
    main()
