import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

print("Attempting to load 'train.csv' and 'test.csv' from the script's directory.")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("\nERROR: 'train.csv' or 'test.csv' not found.")
    print("Exiting program.")
    print("\n"+"="*40)
    sys.exit()

print("\n"+"="*40)
print("\nTraining data loaded successfully.")
print(f"Shape of training data: {train_df.shape}")

print("\nTest data loaded successfully.")
print(f"Shape of test data: {test_df.shape}")

print("\n"+"="*40)
print("\nMissing values in training data:")
print(train_df.isnull().sum().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum().sum())
print("\n"+"="*40)
print("\n Performing Feature Engineering")
for df in [train_df, test_df]:
    df['Rhythmic_Energy'] = df['RhythmScore'] * df['Energy']
    df['Loudness_x_Energy'] = df['AudioLoudness'] * df['Energy']
    if 'TrackDurationMs' in df.columns:
        df['TrackDurationSec'] = df['TrackDurationMs'] / 1000
        df.drop('TrackDurationMs', axis=1, inplace=True)

print("\n"+"="*40)
print("\nData After Feature Engineering ")
print("New columns like 'Rhythmic_Energy' have been added. Here's a look at the first 5 rows:")
print(train_df.head())


TARGET = 'BeatsPerMinute'
FEATURES = [col for col in train_df.columns if col not in ['id', TARGET]]

X_train_full = train_df[FEATURES]
y_train_full = train_df[TARGET]
X_test = test_df[FEATURES]

X_test = X_test[X_train_full.columns]

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

print("\nGenerating Polynomial Features")
print("This will capture more complex interactions between features")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_full_poly = poly.fit_transform(X_train_full_scaled)
X_test_poly = poly.transform(X_test_scaled)
print(f"Feature space expanded from {X_train_full_scaled.shape[1]} to {X_train_full_poly.shape[1]} features.")

X_train, X_val, y_train, y_val = train_test_split(X_train_full_poly, y_train_full, test_size=0.2, random_state=42)

print(f"\nData split into training ({X_train.shape[0]} rows) and validation ({X_val.shape[0]} rows) sets.")

X_train_full.hist(bins=30, figsize=(16, 12), layout=(4, 4))
plt.suptitle("Feature Distributions", y=1.02)
plt.tight_layout()
plt.savefig("feature_distributions.png")
print("\nSaved feature distribution plot to 'feature_distributions.png'")
plt.close()

plt.figure(figsize=(12, 10))
correlation_matrix = train_df[FEATURES + [TARGET]].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
plt.title('Feature Correlation Matrix')
plt.savefig("feature_correlation_heatmap.png")
print("Saved feature correlation heatmap to 'feature_correlation_heatmap.png'")
plt.close()

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, objective='reg:squarederror')
}

results = {}
print("\n"+"="*40)
print("\nTraining and Evaluating Models")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    results[name] = {'MAE': mae, 'RMSE': rmse}
    print(f"  {name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

results_df = pd.DataFrame(results).T
print("\n Model Performance Comparison")
print(results_df)

best_model_name = results_df['RMSE'].idxmin()
best_model = models[best_model_name]
print(f"\n Best performing model: {best_model_name}")

print("\n Why the Best Model Performs Better ")
explanation = """
1. Linear Regression (Worst Performer): This model is the simplest. It tries to find a single straight-line relationship between the features and the BPM. The relationship in this data is far too complex for such a simple model, leading to the highest error.

2. Random Forest (Good Performer): This model is much more flexible. It builds hundreds of individual decision trees on different parts of the data and averages their predictions. This 'committee' approach allows it to capture complex, non-linear patterns that Linear Regression misses.

3. Gradient Boosting & XGBoost (Best Performers): These are the most powerful models here. They build trees sequentially, where each new tree is specifically trained to correct the errors made by the previous ones. This focused, error-correcting process allows them to learn the most subtle and difficult patterns in the data, resulting in the lowest error. XGBoost is an optimized version of this technique.
"""
print(explanation)

print("\n Training Final Model")
print(f"Retraining the best model ({best_model_name}) on the full dataset")
best_model.fit(X_train_full_poly, y_train_full)

print("\n Making Final Predictions")
final_predictions = best_model.predict(X_test_poly)

submission_df = pd.DataFrame({'ID': test_df['id'], 'BeatsPerMinute': final_predictions})
submission_df['BeatsPerMinute'] = submission_df['BeatsPerMinute'].clip(lower=0)
submission_df.to_csv('submission.csv', index=False)

print("\n 'submission.csv' created successfully")
