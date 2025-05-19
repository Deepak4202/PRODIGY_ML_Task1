# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the training dataset
train_df = pd.read_csv("train.csv")

# Separate numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['number']).columns
categorical_cols = train_df.select_dtypes(exclude=['number']).columns

# Remove 'SalePrice' from the list of numerical features
numerical_features = [col for col in numerical_cols if col != 'SalePrice']

# Impute missing values in numerical columns using mean
imputer = SimpleImputer(strategy='mean')
train_df[numerical_features] = imputer.fit_transform(train_df[numerical_features])

# Final dataframe for modeling
train_df = train_df[numerical_features + ['SalePrice']]

# Features (X) and Target (y)
X = train_df[numerical_features]
y = train_df['SalePrice']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test split
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluation on train.csv test split:")
print("Mean Squared Error: {:.2f}".format(mse))
print("R-squared: {:.4f}".format(r2))

# --- Predict on external test.csv ---
external_test_df = pd.read_csv("test.csv")

# Use only the columns used in training
external_test_data = external_test_df[numerical_features]

# Impute missing values in external test set
external_test_data = imputer.transform(external_test_data)

# Predict
external_predictions = model.predict(external_test_data)

# Save predictions to CSV
output_df = external_test_df.copy()
output_df['Predicted_SalePrice'] = external_predictions
output_df.to_csv("predictions.csv", index=False)
print("\nPredictions saved to predictions.csv")

# --- Visualizations ---

# 1. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
