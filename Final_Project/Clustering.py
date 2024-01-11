import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns

data=pd.read_excel("D:/Programming/Marketing/CleansData.xlsx")
features=['Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness']
data=data.dropna()

n_cluster=3
model = KMeans(n_cluster,random_state=1)
model.fit(data[features].values)
data['cluster'] = model.fit_predict(data[features].values)
print(data)


for i in range(n_cluster):
    cluster_data = data.loc[data['cluster'] == i] 
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=cluster_data, x=feature, bins=5, kde=True)
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.title(f'Cluster {i+1}: {feature} Histogram')
        plt.show()