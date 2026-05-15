import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load and clean data
df = pd.read_csv("./data_scrapper/divar_tara_ml_dataset.csv")

cols_to_drop = ['title', 'token', 'url', 'description_snippet', 'brand_model', 'year']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df = df.dropna(subset=['price_toman'])

# Remove Outliers (Noise)
Q1 = df['price_toman'].quantile(0.25)
Q3 = df['price_toman'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['price_toman'] >= lower_bound) & (df['price_toman'] <= upper_bound)]
df = df.fillna(0)

# Convert categories to numbers
cat_cols = ['color', 'body_status', 'engine_status', 'chassis_status', 'gearbox', 'fuel_type', 'airbag', 'tire_quality']
df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)

X = df_encoded.drop(columns=['price_toman']).values
y = df_encoded['price_toman'].values.reshape(-1, 1)

# 2. Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalize data (Crucial for Neural Networks)
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

# 4. Build Keras Neural Network
model = Sequential()

# Input layer and First hidden layer (16 neurons)
model.add(Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'))

# Second hidden layer (8 neurons)
model.add(Dense(8, activation='relu'))

# Output layer (1 neuron for price prediction)
model.add(Dense(1, activation='linear'))

# 5. Compile model
model.compile(optimizer='adam', loss='mse')

# 6. Train the model
print("Training Keras Model...")
model.fit(X_train_scaled, y_train_scaled, epochs=800, batch_size=4, verbose=1)

# 7. Evaluate on Test Data
y_pred_scaled = model.predict(X_test_scaled)

# Convert scaled prices back to real Toman
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test, y_pred_real)

print("\n--- Final Result (Keras NN) ---")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):,.0f} Toman")