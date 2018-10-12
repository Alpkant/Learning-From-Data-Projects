import numpy as np
import matplotlib.pyplot as plt
import random

dataset   = np.loadtxt('data.txt', delimiter=',', usecols=range(0,64))
labels    = np.loadtxt('data.txt', delimiter=',', dtype=np.str , usecols=[64])

cov_mat = np.cov(dataset,rowvar=False)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print(np.shape(eig_vecs))
reduced = []
for vec in eig_vecs:
    reduced.append(vec[0:2])

reduced = np.array(reduced).T

x = []
y = []
for i in range(len(dataset)):
    temp = np.dot(reduced, dataset[i].T)    #Find the x and y values
    x.append(temp[0])
    y.append(temp[1])


plt.scatter(x,y,s=1)
samples = random.sample(range(0,len(x)),200)   #Take random 200 points
for i in samples:
    plt.annotate(labels[i],(x[i],y[i])) #Draw classes of the points

plt.xlabel("First eigenvector")
plt.ylabel("Second eigenvector")
plt.show()
