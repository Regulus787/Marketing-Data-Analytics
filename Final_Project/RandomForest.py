import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score,recall_score,r2_score
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_excel("D:/Programming/Marketing/CleansData.xlsx")
features=['Gender', 'Customer Type', 'Age', 'Type of Travel',
       'Class', 'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

data=data.dropna()

X=data[features]
y=data['satisfaction']
print(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=20)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

MSE = mean_squared_error(y_true = y_test, y_pred = y_prediction)
accuracy=accuracy_score(y_test,y_prediction)
precision=precision_score(y_test,y_prediction)
recall=recall_score(y_test,y_prediction)
r2=r2_score(y_test,y_prediction)
print("MSE:",MSE)
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("R2:",r2)

importances = model.feature_importances_

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

print(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()


