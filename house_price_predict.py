import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Bengaluru_House_Data.csv")

# Extract number of bedrooms
df['bedrooms'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]) if pd.notnull(x) and str(x).split(' ')[0].isdigit() else np.nan)

# Convert total_sqft to float
def convert_sqft(sqft):
    try:
        if '-' in str(sqft):
            tokens = sqft.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(sqft)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

# Clean data
data = df[['total_sqft', 'bedrooms', 'bath', 'price']].dropna()
X = data[['total_sqft', 'bedrooms', 'bath']]
y = data['price']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:")
print(f"total_sqft coef: {model.coef_[0]:.4f}")
print(f"bedrooms coef: {model.coef_[1]:.4f}")
print(f"bath coef: {model.coef_[2]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# --------- Plot 1: Actual vs Predicted (Red) ---------
plt.figure(figsize=(8, 6), dpi=100)
plt.scatter(y_test, y_pred, color='red',marker='*')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig('d:/Virtual Internship/output.png')
plt.close()

# --------- Plot 2: Best-Fit Line ---------
avg_bed = int(X['bedrooms'].mean())
avg_bath = int(X['bath'].mean())

sqft_range = np.linspace(X['total_sqft'].min(), X['total_sqft'].max(), 100)
input_features = np.column_stack((sqft_range, [avg_bed]*100, [avg_bath]*100))
predicted_prices = model.predict(input_features)

plt.figure(figsize=(8, 6), dpi=100)
subset = data.sample(500)
# Blue stars for actual data
plt.scatter(subset['total_sqft'], subset['price'], label="Actual Data", color='blue', marker='*')
# Green line for best fit
plt.plot(sqft_range, predicted_prices, color='green', linewidth=2, label="Best Fit Line")
plt.xlabel("Total Sqft")
plt.ylabel("Price (Lakhs)")
plt.title("Best Fit Line: Total Sqft vs Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('d:/Virtual Internship/best_fit_line.png')
plt.close()
