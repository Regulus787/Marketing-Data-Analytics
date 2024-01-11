import pandas as pd
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("D:/Programming/Marketing/train.csv")
test=pd.read_csv("D:/Programming/Marketing/train.csv")
data=pd.concat([train,test],axis=0)
data = data.drop(data.columns[[0, 1]], axis=1)
data=data.dropna()
data['Gender'],gender_label=data['Gender'].factorize()
data['Customer Type'],customer_label=data['Customer Type'].factorize()
data['Type of Travel'],Travel_type_label=data['Type of Travel'].factorize()
data['Class'],class_label=data['Class'].factorize()
data['satisfaction'],satisfaction_label=data['satisfaction'].factorize()
features = ['Age','Flight Distance','Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink','Online boarding','Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service','Baggage handling', 'Checkin service', 'Inflight service','Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

#print(data)
data.to_excel("D:/Programming/Marketing/CleansData.xlsx")

print(satisfaction_label)