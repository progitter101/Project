"""
train_model.py
Trains a Random Forest model on the California Housing dataset
(loaded from housing.csv) and saves it as house_price_model.pkl
for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Data collection
df = pd.read_csv('housing.csv')

# 2. Data cleaning
df = df.drop_duplicates()
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# 3. Feature engineering (same style as sklearn's built-in dataset)
df['AveRooms'] = df['total_rooms'] / df['households']
df['AveBedrms'] = df['total_bedrooms'] / df['households']
df['AveOccup'] = df['population'] / df['households']

feature_cols = [
    'median_income', 'housing_median_age', 'AveRooms', 'AveBedrms',
    'population', 'AveOccup', 'latitude', 'longitude'
]
X = df[feature_cols]
y = df['median_house_value']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model training
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Model trained successfully.")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# 7. Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# 8. Save model
joblib.dump(model, 'house_price_model.pkl')
print("\nModel saved as house_price_model.pkl")
