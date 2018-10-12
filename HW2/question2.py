import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import scipy.special as sc
from sklearn.metrics import confusion_matrix
from sys import exit

fid = open("iris_data.txt", "r")
li = fid.readlines()
fid.close()
shuffle(li)
fid = open("iris_data.txt","w")
fid.writelines(li)
fid.close()
#I shuffled the data set since it was sorted data and this will be bad
#if i use 10 fold cross validation because some test sets will only include same class

x   = np.loadtxt('iris_data.txt', delimiter=',', usecols=[0,1,2,3], dtype=np.float128)
x   = np.hstack((np.ones((np.shape(x)[0],1)),x))    #Add first column with full of ones for x0 value
labels = np.loadtxt('iris_data.txt', delimiter=',', dtype=np.str , usecols=[4]) #Save the class labels
d = np.shape(x)[1] #Number of feature including x0
N = np.size(labels)   #Number of training input
K = 3   #Number of class

y_predict = []  #My prediction class will be populated inside tenfoldvalidation function



def softmax(vec) :#I am creating vectorized version of softmax function
    e_x = np.exp(vec - np.max(vec)) #In order to avoid the overflow i substrac max of vec
    return e_x / e_x.sum(axis=0)

def tenfoldvalidation(rate):
    #I am splitting and concatanate the dataset according to the 10 fold
    global K,N,w,y,y_predict,labels,x
    splittedsets   = np.split(x,10)
    splittedlabels = np.split(labels,10)

    for testSet in range(10):   #10 different test
        train_data_set        = np.array([])
        train_data_set_labels = np.array([])
        for i in range(10):
            if i==testSet:  #Do not include test set to train data
                continue
            else:
                train_data_set        = np.vstack([train_data_set,splittedsets[i]]) if train_data_set.size else splittedsets[i]
                train_data_set_labels = np.concatenate((train_data_set_labels, splittedlabels[i])) if train_data_set_labels.size else splittedlabels[i]

        r = np.zeros((np.shape(train_data_set)[0],K)) #Expected r
        for i in range(np.shape(train_data_set)[0]):
            if train_data_set_labels[i] == 'Iris-setosa':
                r[i][0] = 1
            elif train_data_set_labels[i] =='Iris-versicolor':
                r[i][1] = 1
            else:
                r[i][2] = 1

        TrainTheDataSet(train_data_set,np.shape(train_data_set)[0],r,train_data_set_labels,splittedsets[testSet],splittedlabels[testSet],rate)
        print("Fold " ,testSet+1 , " is finished.")
    plot_confusion_matrix(y_predict,labels) #10 test is finished we have predictions now. Plot the confusion matrix

def TrainTheDataSet(x,N,r,y_reallabels,testSet,testLabel,rate):

    global K,d,w,y_predict
    w = np.random.uniform(low=np.float128(-0.1), high=np.float128(0.1), size=(K,d)) #Random initialize
    diffw = np.zeros((K,d))    #Differences of wij = 0
    o = np.zeros((1,K), dtype=np.float128) # o values
    y = np.zeros((N,K))

    for converge in range(400):
        diffw = np.zeros((K,d)) # Assign 0 for every loop

        for t in range(0,N):
            for i in range(0,K):
                o[0][i] = 0
                for j in range(d):
                    o[0][i] += np.dot(w[i][j],x[t][j])


            y[t] = softmax(o[0]) #There is no for so it's vectorized and calculates for K classes at once


            for i in range(0,K):
                for j in range(0,d):
                    diffw[i][j] += (r[t][i] - y[t][i])*x[t][j]

        for i in range(0,K):
            for j in range(0,d):
                w[i][j] += rate*diffw[i][j]
        print(diffw,"\n")

    #In this part training finished and i am testing the test data set and will print accuracy
    o = np.zeros((1,K), dtype=np.float128)
    y = np.zeros((N,K))
    for k in range(np.shape(testSet)[0]):
        for i in range(0,K):
            o[0][i] = 0
            for j in range(d):
                o[0][i] += np.dot(w[i][j],testSet[k][j])

        y[k] = softmax(o[0])
        indices = np.where(y[k] == y[k].max()) #This code finds the biggest number's indice in the 1x3 vector

        if y[k][indices[0]] == 1 and r[k][indices[0]]==1:   #If answer is true
            y_predict.append(testLabel[k])
        else:                                               #If prediction is wrong
            if indices[0] == 0:
                y_predict.append('Iris-setosa')
            elif indices[0] == 1:
                y_predict.append('Iris-versicolor')
            else :
                y_predict.append('Iris-virginica')


def plot_confusion_matrix(y_predict,labels):
    cm = confusion_matrix(labels,y_predict)
    numberOfTrue = np.trace(np.asarray(cm))
    totalLabel   = cm.sum()
    accuracyString = "Accuracy is :" + str(100*numberOfTrue/totalLabel) + "% True: " + str(numberOfTrue) + " False:" + str(totalLabel-numberOfTrue)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.suptitle(accuracyString)
    plt.colorbar()
    plt.ylabel('Real class label')
    plt.xlabel('Predicted label')
    plt.show()

#Main function
def main():
    tenfoldvalidation(rate=1)

if __name__ == "__main__":
    main()
