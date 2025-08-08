Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> # Mining Process Quality Prediction - Final Project Code
... 
... import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... import seaborn as sns
... from sklearn.model_selection import train_test_split
... from sklearn.ensemble import RandomForestRegressor
... from sklearn.metrics import mean_squared_error, mean_absolute_error
... 
... # Load dataset
... df = pd.read_csv("/mining_process_quality_sample.csv")
... 
... # Check for nulls or issues
... print("Null values in dataset:\n", df.isnull().sum())
... 
... # Define features and target
... features = ['Temperature', 'Pressure', 'Oxygen_Flow', 'Nitrogen_Flow', 'Fe_Content']
... target = 'Silica_Percent'
... 
... X = df[features]
... y = df[target]
... 
... # Train-test split
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... # Train model
... model = RandomForestRegressor(n_estimators=100, random_state=42)
... model.fit(X_train, y_train)
... 
... # Predict and evaluate
... y_pred = model.predict(X_test)
... rmse = np.sqrt(mean_squared_error(y_test, y_pred))
... mae = mean_absolute_error(y_test, y_pred)
... 
print("Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label='Actual', marker='o')
plt.plot(y_pred[:100], label='Predicted', marker='x')
plt.title("Actual vs Predicted Silica Percentage")
plt.xlabel("Sample")
plt.ylabel("Silica Percent")
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features)
feat_imp.sort_values().plot(kind='barh', title='Feature Importance')
plt.tight_layout()
plt.show()

# Save model (optional)
#import joblib
#joblib.dump(model, 'silica_prediction_model.pkl')
