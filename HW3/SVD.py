import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy.misc
from PIL import Image

def SVD(A):
    aut = np.dot(A.T,A)
    tmp = np.linalg.eig(aut)
    eigvals = tmp[0]
    V = tmp[1]                  #Eigen vectors with normalization
    eigvals[::-1].sort()        #Sort eigen values in descending order

    S = np.zeros((len(eigvals),len(eigvals)))
    for i in range(len(eigvals)):
        S[i][i] = math.sqrt(np.absolute(eigvals[i]))

    S_inv = np.linalg.inv(S)
    U = np.dot(np.dot(A,V),S_inv)   #Calculate U vector

    return U,S,V


def create_image(U,S,V,rank=100):
    rank = int (rank * np.shape(S)[0]  /100)
    S_rank = np.zeros((rank,rank))
    for i in range(rank):
        S_rank[i][i] = S[i][i]

    U_rank = U[:,:rank]
    V_rank = V[:,:rank]
    A = np.dot(np.dot(U_rank,S_rank),V_rank.T)  #Create compressed matrix according to the rankı value
    return A

rawimage = scipy.misc.imread("data.jpg")
red_channel   = rawimage[:,:,0]
green_channel = rawimage[:,:,1]
blue_channel  = rawimage[:,:,2]


U_red  , S_red  , V_red   = SVD(red_channel)
U_green, S_green, V_green = SVD(green_channel)
U_blue , S_blue , V_blue  = SVD(blue_channel)


rgbArray = np.zeros( np.shape(red_channel) + (3,), 'uint8')
rank = 80
rgbArray[:, :, 0] = create_image(U_red,S_red,V_red,rank)
rgbArray[: , :, 1] = create_image(U_green,S_green,V_green,rank)
rgbArray[: , :, 2] = create_image(U_blue,S_blue,V_blue,rank)
img = Image.fromarray(rgbArray)
img.save('output.jpg')
