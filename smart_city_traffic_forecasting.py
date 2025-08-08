# Smart City Traffic Forecasting - Final Project Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

# Load dataset (replace with actual path if needed)
df = pd.read_csv("/smart_city_traffic_sample.csv")

# Convert date_time to datetime format
df['date_time'] = pd.to_datetime(df['date_time'])

# Feature Engineering
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month

# Example: Forecasting for junction_1 (can repeat for others)
target = 'junction_1'
features = ['hour', 'day_of_week', 'month']

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Performance for {target}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Plotting actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label='Actual', marker='o')
plt.plot(y_pred[:100], label='Predicted', marker='x')
plt.title(f"Actual vs Predicted Traffic at {target}")
plt.xlabel("Sample")
plt.ylabel("Traffic Volume")
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importance
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features)
feat_imp.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Save model (optional)
#import joblib
#joblib.dump(model, 'junction1_traffic_model.pkl')
