# -----------------------------
# Multiple Linear Regression Project
# Dataset: economic_index.csv
# -----------------------------

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

# -----------------------------
# 2. Load Dataset
# -----------------------------
df_index = pd.read_csv("economic_index.csv")
print(df_index.head())

# -----------------------------
# 3. Drop unnecessary columns
# -----------------------------
df_index.drop(columns=['Unnamed: 0', 'year', 'month'], inplace=True)
print(df_index.head())

# -----------------------------
# 4. Check for null values
# -----------------------------
print(df_index.isnull().sum())

# -----------------------------
# 5. Visualize Data
# -----------------------------
sns.pairplot(df_index)
plt.show()

# Correlation matrix
print("Correlation matrix:\n", df_index.corr())

# Scatter plot examples
plt.scatter(df_index['interest_rate'], df_index['unemployment_rate'], color='red')
plt.xlabel('Interest Rate')
plt.ylabel('Unemployment Rate')
plt.title('Interest Rate vs Unemployment Rate')
plt.show()

plt.scatter(df_index['interest_rate'], df_index['index_price'], color='blue')
plt.xlabel('Interest Rate')
plt.ylabel('Index Price')
plt.title('Interest Rate vs Index Price')
plt.show()

# -----------------------------
# 6. Prepare Independent and Dependent Variables
# -----------------------------
# Method 1: Using column names
X = df_index[['interest_rate', 'unemployment_rate']]
y = df_index['index_price']

# Method 2: Using iloc (alternative)
# X = df_index.iloc[:, :-1]
# y = df_index.iloc[:, -1]

# -----------------------------
# 7. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# 8. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 9. Multiple Linear Regression (sklearn)
# -----------------------------
regression = LinearRegression()
regression.fit(X_train, y_train)

# Coefficients and Intercept
print("Coefficients:", regression.coef_)
print("Intercept:", regression.intercept_)

# -----------------------------
# 10. Cross-Validation
# -----------------------------
cv_scores = cross_val_score(
    estimator=regression,
    X=X_train,
    y=y_train,
    scoring='neg_mean_squared_error',
    cv=5
)
print("Cross-validation MSE scores:", -cv_scores)
print("Mean CV MSE:", -np.mean(cv_scores))

# -----------------------------
# 11. Predictions
# -----------------------------
y_pred = regression.predict(X_test)

# Prediction metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]
k = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n-1)/(n-k-1)

print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)

# -----------------------------
# 12. Residual Analysis
# -----------------------------
residuals = y_test - y_pred

# Scatter plot y_test vs y_pred
plt.scatter(y_test, y_pred, color='green')
plt.xlabel('Actual Index Price')
plt.ylabel('Predicted Index Price')
plt.title('Actual vs Predicted')
plt.show()

# Residual distribution
sns.kdeplot(residuals, fill=True)
plt.title('Residuals Distribution')
plt.show()

# Residuals vs Predictions
plt.scatter(y_pred, residuals, color='orange')
plt.xlabel('Predicted Index Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()

# -----------------------------
# 13. OLS Regression (statsmodels)
# -----------------------------
X_train_ols = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_ols).fit()
print(ols_model.summary())

# Compare coefficients with sklearn model
print("Sklearn coefficients:", regression.coef_)
print("Sklearn intercept:", regression.intercept_)

