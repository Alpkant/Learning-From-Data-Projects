import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import seaborn as sns

def main ():
    k= int(input("How many clusters do you want ?"))

    #Read the data without headers
    dataset   = np.genfromtxt('Cluster.csv', delimiter=',' , usecols=range(1,65) , skip_header=1)
    #Initialize the PCA instance
    pca = PCA(n_components=2) # 2 features
    decomposed = pca.fit_transform(dataset)
    feature1   = decomposed[:,:1]
    feature2   = decomposed[:,1:2]

    K_Means(feature1,feature2,k)

# Squared of the Ecludian distance
def dist(xa,ya,xb,yb):
    return ((xa-xb) ** 2) + ((ya-yb) ** 2)

# Compare if two subsequent cluster means are the same
def difference(C_x,C_y,C_x_old,C_y_old):
    return (np.array_equal(C_x,C_x_old) and np.array_equal(C_y,C_y_old))

#If you want to get sumOfSquareerror than you should call this function after everything is
# calculated so in this file just before plotting
def sumOfSquareError(CX,CY,Feature1,Feature2,C_values,k):
    sumOfSquareError = 0.0
    for i in range(len(Feature1)):
        j = C_values[i]
        sumOfSquareError+= dist(CX[j],CY[j],Feature1[i],Feature2[i])

    print("For " ,k," cluster the error is: ",sumOfSquareError)
    return sumOfSquareError

def K_Means(feature1,feature2,k=2):
    #Randomly generate k initial clusters
    C_x = np.random.randint(np.amin(feature1)+1, np.amax(feature1)-2, size=k).astype("float")
    C_y = np.random.randint(np.amin(feature2)+1, np.amax(feature2)-2, size=k).astype("float")
    C_x_old = np.zeros(np.shape(C_x))
    C_y_old = np.zeros(np.shape(C_y))

    #It will keep every point's assigned cluster like 0,1,2
    C_values = np.zeros(len(feature1),dtype=int)

    while not difference(C_x,C_y,C_x_old,C_y_old): #While clusters are moving
        counter = np.zeros(k) #It keeps how many point's each cluster has

        for i in range(len(feature1)):
            cluster_counter = 0
            min = float('inf')

            for j in range(len(C_x)):
                distance = dist(feature1[i],feature2[i],C_x[j],C_y[j])
                if distance < min : #If point is close to this cluster
                    min = distance
                    cluster_counter=j

            C_values[i] = cluster_counter
            counter[cluster_counter] += 1

        # Find mean values of the each cluster for updating
        sumTemp_x = np.zeros(k)
        sumTemp_y = np.zeros(k)
        for j in range(len(C_values)):
            sumTemp_x[C_values[j]] += feature1[j]
            sumTemp_y[C_values[j]] += feature2[j]

        #Keep clusters of the one computation before and update the new cluster points
        for j in range(k):
            sumTemp_x[j] = np.divide(sumTemp_x[j],counter[j])
            sumTemp_y[j] = np.divide(sumTemp_y[j],counter[j])
            C_x_old[j] = C_x[j]
            C_y_old[j] = C_y[j]
            C_x[j] = sumTemp_x[j]
            C_y[j] = sumTemp_y[j]
    #End of while

    sumOfSquareError(C_x,C_y,feature1,feature2,C_values,k)
    #Print the classes according to their classes
    colors = ['b', 'g', 'r', 'c', 'm', 'y','k',sns.xkcd_rgb["olive drab"],sns.xkcd_rgb["barbie pink"],sns.xkcd_rgb["mustard brown"] ]
    for i in range(k):
        points = np.array([(feature1[j],feature2[j]) for j in range(len(feature1)) if C_values[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

    plt.scatter(C_x, C_y, marker='*', s=200,color='k') #Clusters
    plt.show()

main()
