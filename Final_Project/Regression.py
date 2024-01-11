import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_prediction=model.predict(X_test)

mse=mean_squared_error(y_prediction,y_test)
r2=r2_score(y_prediction,y_test)
accuracy=accuracy_score(y_prediction,y_test)
print("mse",mse)
print("R2",r2)
print("Accuracy",accuracy)

coefficients = model.coef_
feature_names = X.columns

print("Coefficients:")
for feature, coef in zip(feature_names, coefficients[0]):
    print(f"{feature}: {coef}")

print("Intercept:", model.intercept_)

