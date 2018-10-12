

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
# Opening the file
df = pd.read_csv('train.txt', sep=",")

print(df.groupby(['creationDate']).mean()['returnShipment'])