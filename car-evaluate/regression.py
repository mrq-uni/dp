import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load data
df = pd.read_csv("./data_scrapper/divar_tara_ml_dataset.csv")

# 2. Drop useless columns ('year' is dropped because we have 'age')
columns_to_drop = ['title', 'token', 'url', 'description_snippet', 'brand_model', 'year']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Drop rows with no price
df = df.dropna(subset=['price_toman'])

# 3. Find and remove Outliers (Noise) using IQR method
Q1 = df['price_toman'].quantile(0.25)
Q3 = df['price_toman'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers to show them before removing
outliers = df[(df['price_toman'] < lower_bound) | (df['price_toman'] > upper_bound)]

print("==================================================")
print(f"Found outliers (noise): {len(outliers)} records")
if len(outliers) > 0:
    print("Prices of these outliers (Toman):")
    print(outliers['price_toman'].values)
print("Note: These noise records are successfully removed from the dataset.")
print("==================================================\n")

# Keep only normal prices (Remove noise)
df = df[(df['price_toman'] >= lower_bound) & (df['price_toman'] <= upper_bound)]

# Fill missing values with 0
df = df.fillna(0)

# 4. Convert text categories to numbers (One-Hot Encoding)
categorical_features = ['color', 'body_status', 'engine_status', 'chassis_status', 'gearbox', 'fuel_type', 'airbag', 'tire_quality']
df_encoded = pd.get_dummies(df, columns=[col for col in categorical_features if col in df.columns], drop_first=True)

# 5. Prepare data for the model
X = df_encoded.drop(columns=['price_toman'])
y = df_encoded['price_toman']

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("--- Linear Regression Results ---")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):,.0f} Toman")