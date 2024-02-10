

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


data = pd.read_csv("fifa_players.csv")

print(data['value_euro'].isnull().sum())
mean_value = data['value_euro'].mean()
data['value_euro'].fillna(mean_value, inplace=True)


X = data[['age', 'height_cm', 'weight_kgs', 'overall_rating', 'potential', 'international_reputation(1-5)', 
          'weak_foot(1-5)', 'skill_moves(1-5)']]
y = data['value_euro']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

sample_data = [[25, 180, 75, 85, 90, 4, 4, 4]]
sample_prediction = model.predict(sample_data)
print('Sample Prediction:', sample_prediction)

coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print(coefficients)
