import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression


def clearData(data):
    # Taking reasonable data
    allDataFirst = [d for d in zip(data['orderDate'], data['deliveryDate']) if list(d)[1] != "?"  ]
    allDataAll = [list(d) for d in allDataFirst if datetime.strptime(list(d)[0], '%Y-%m-%d') < datetime.strptime(list(d)[1], '%Y-%m-%d')]

    # Calculating ship time
    shipTime = [(datetime.strptime(d[1], '%Y-%m-%d')-datetime.strptime(d[0], '%Y-%m-%d')) / timedelta (days=1) for d in allDataAll]
    shiptimeMean = np.mean(np.array(shipTime))

    # First changin ? with orderDate + Mean
    clearedDataFirst = [list(d) if list(d)[1] != "?" else [list(d)[0],(datetime.strptime(list(d)[0], '%Y-%m-%d') + timedelta(days=11)).strftime("%Y-%m-%d") ]  for d in zip(data['orderDate'], data['deliveryDate']) ]

    # Second changing 1990 with orderDate + Mean
    clearedTimeDataSecond = [d if datetime.strptime(list(d)[0], '%Y-%m-%d') < datetime.strptime(list(d)[1], '%Y-%m-%d') else [d[0],(datetime.strptime(list(d)[0], '%Y-%m-%d') + timedelta(days=11)).strftime("%Y-%m-%d")] for d in clearedDataFirst]

    # Changing data with new values
    data["orderDate"] = [row[0] for row in clearedTimeDataSecond]
    data["deliveryDate"] = [row[1] for row in clearedTimeDataSecond]

    # Calculating age mean
    allDataFirstAge = [d for d in data['dateOfBirth'] if d != "?"  ]
    allDataAllAge = [d for d in allDataFirstAge if datetime.strptime(d, '%Y-%m-%d') > datetime.strptime("1930-01-01", '%Y-%m-%d') and datetime.strptime(d, '%Y-%m-%d') < datetime.strptime("2000-01-01", '%Y-%m-%d')]
    ageYear = [((datetime.strptime("2014-01-01", '%Y-%m-%d')-datetime.strptime(d, '%Y-%m-%d'))/timedelta (days=1))/365 for d in allDataAllAge]
    ageYearMean = np.mean(np.array(ageYear))

    # Clearing the birthdate data
    allDataFirstAgeChanged = [d if d != "?" else (datetime.strptime("2014-01-01", '%Y-%m-%d') - timedelta(days=48)).strftime('%Y-%m-%d') for d in data['dateOfBirth']   ]
    allDataAllAgeChanged = [d if datetime.strptime(d, '%Y-%m-%d') > datetime.strptime("1930-01-01", '%Y-%m-%d') and datetime.strptime(d, '%Y-%m-%d') < datetime.strptime("2000-01-01", '%Y-%m-%d') else (datetime.strptime("2014-01-01", '%Y-%m-%d') - timedelta(days=48*365.2)).strftime('%Y-%m-%d') for d in allDataFirstAgeChanged]
    data["dateOfBirth"] = allDataAllAgeChanged

    colorDict = {}
    colorArary = ['denim', 'ocher', 'curry', 'green', 'black', 'brown', 'red', 'mocca',
     'anthracite', 'olive', 'petrol', 'blue', 'grey', 'beige', 'ecru', 'turquoise',
     'magenta', 'purple', 'pink', 'khaki', 'navy', 'habana', 'silver', 'white',
     'nature', 'stained', 'orange', 'azure', 'apricot', 'mango', 'berry', 'ash',
     'hibiscus', 'fuchsia', 'dark denim', 'mint', 'ivory', 'yellow',
     'bordeaux', 'pallid', 'ancient', 'baltic blue', 'almond', 'aquamarine',
     'aubergine','aqua', 'dark garnet', 'dark grey', 'avocado', 'creme',
     'champagner', 'cortina mocca', 'currant purple', 'cognac', 'aviator', 'gold',
     'ebony', 'cobalt blue', 'kanel','curled', 'caramel', 'antique pink',
     'darkblue', 'copper coin', 'terracotta', 'basalt', 'amethyst', 'coral', 'jade',
     'opal', 'striped', 'mahagoni', 'floral', 'dark navy', 'dark oliv',
     'vanille', 'ingwer', 'iron', 'graphite', 'leopard', 'bronze', 'crimson',
     'lemon', 'perlmutt']

    for index, x in enumerate(colorArary):
        colorDict[x] = index
    colorData = list(data['color'])
    for index, x in enumerate(colorData):
        if x  == "?":
            colorData[index] = "black"
        elif x  == "brwon":
            colorData[index] = "brown"
        elif x  == "blau":
            colorData[index] = "blue"
        elif x  == "oliv":
            colorData[index] = "olive"
        colorData[index] = colorDict[colorData[index]]
    data['color'] = colorData
    return data

data = pd.read_csv('train.txt', sep=",")

data = clearData(data)

# Feature 1, if taken date > 10 1, else 0
shipmentDateF = np.array([ 0 if ((datetime.strptime(list(d)[1], '%Y-%m-%d')-datetime.strptime(list(d)[0], '%Y-%m-%d')) / timedelta (days=1)) <= 10 else 1 for d in zip(data['orderDate'], data['deliveryDate'])])

# Feature 2 - specialDateChrismass, if order is close to speacial date and came late it is 1
# Feature 3 - specialDateOthers, if order is close to speacial date and came late it is 1
dateRepresentationOfData = [ [(int('%02d' % datetime.strptime(list(d)[0], '%Y-%m-%d').day),int('%02d' % datetime.strptime(list(d)[0], '%Y-%m-%d').month)), (int('%02d' % datetime.strptime(list(d)[1], '%Y-%m-%d').day),int('%02d' % datetime.strptime(list(d)[1], '%Y-%m-%d').month))] for d in zip(data['orderDate'], data['deliveryDate'])]
specialDateChrismassF = np.array([1 if (d[0][0] > 14 and d[0][0] < 22 and d[0][1] == 12) and (d[1][0] >= 1 and d[1][1] >= 1) else 0 for d in dateRepresentationOfData])
specialDateOthersF = np.array([1 if (d[0][0] > 3 and d[0][0] < 11 and d[0][1] == 5) and (d[1][0] >= 15 and d[1][1] >= 5) or (d[0][0] > 8 and d[0][0] < 14 and d[0][1] == 2) and (d[1][0] >= 18 and d[1][1] >= 2) else 0 for d in dateRepresentationOfData])

# Feature 4, <30 - 0 , 30-55 - 1, 55> - 2
ageNummArray = [((datetime.strptime("2014-01-01", '%Y-%m-%d')-datetime.strptime(d, '%Y-%m-%d'))/timedelta (days=1))/365 for d in data['dateOfBirth']]
ageNumF = np.array([0 if d <= 30 else (1 if d <= 55 else 2) for d in ageNummArray])

# Feature 5 manufacturer id mean value
groupedMeanManuReturnRate = data.groupby('manufacturerID').mean()['returnShipment']
manuReturnRateF = np.array([groupedMeanManuReturnRate[d] for d in data["manufacturerID"]])

# Feature 6 costumer id mean value
groupedMeanUserReturnRate = data.groupby('customerID').mean()['returnShipment']
userReturnRateF = np.array([groupedMeanUserReturnRate[d] for d in data["customerID"]])
