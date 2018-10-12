import numpy as np
from datetime import datetime

with open("train.txt", "r") as f:
    data = f.readlines()
    l = [i.split(',')for i in data]
    labels = l[0]
    lNew = l[1:]
    labelsChecker = []
    itemList = []
    dateOfBirth = []
    state = []
    color = []
    price = []
    for indexOne, item in enumerate(lNew):
        itemList.append((int(item[3]), item[4]))
        state.append(item[11])
        color.append(item[5])
        price.append(float(item[7]))
        if(item[10] != "?"):
            dt = datetime.strptime(item[10], '%Y-%m-%d')
            dateOfBirth.append(dt)
        for index, elem in enumerate(item):
            if(elem == "?" or elem == "?\n"):
                labelsChecker.append(labels[index])

    itemListSorted = sorted(itemList)
    with open("order.txt", "w") as f:
        for i in itemListSorted:
            f.write(str(i[0]))
            f.write("\t\t")
            f.write(i[1])
            f.write("\n")
    print(sorted(price)[0])
    print(sorted(price,reverse=True)[0])
    print(sorted(set(color)))
    print(set(state))
    print(sorted(dateOfBirth)[0])
    print("file row", len(l))
    print("? num", len(labelsChecker))
    print(set(labelsChecker))
