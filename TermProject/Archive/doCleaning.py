import numpy as np
import pandas as pd


data = pd.read_csv('train.txt', sep=",")
print(data.dtypes)


# with open("train.txt", "r") as f:
#     data = f.readlines()
#     l = [i.split(',')for i in data]
#     labels = l[1]
#     print(labels)
