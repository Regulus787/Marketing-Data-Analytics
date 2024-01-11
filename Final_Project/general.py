import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("D:/Programming/Marketing/train.csv")
test=pd.read_csv("D:/Programming/Marketing/train.csv")
data=pd.concat([train,test],axis=0)
pd.set_option("display.max_columns", None)
data = data.drop(data.columns[[0, 1]], axis=1)
data=data.dropna()

print(data.groupby('Class').mean())