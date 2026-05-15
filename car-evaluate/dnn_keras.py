import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and clean data (Same as before)
df = pd.read_csv("./data_scrapper/divar_tara_ml_dataset.csv")
cols_to_drop = ['title', 'token', 'url', 'description_snippet', 'brand_model', 'year']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).dropna(subset=['price_toman'])

Q1, Q3 = df['price_toman'].quantile(0.25), df['price_toman'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price_toman'] >= Q1 - 1.5 * IQR) & (df['price_toman'] <= Q3 + 1.5 * IQR)].fillna(0)

cat_cols = ['color', 'body_status', 'engine_status', 'chassis_status', 'gearbox', 'fuel_type', 'airbag', 'tire_quality']
df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)

X = df_encoded.drop(columns=['price_toman']).values
y = df_encoded['price_toman'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Normalization
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

# ==========================================
# 3. Build Deep Neural Network (DNN)
# ==========================================
model = Sequential()

# Input Layer & Hidden Layer 1 (32 Neurons)
model.add(Dense(32, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2)) # Turn off 20% of neurons to prevent overfitting

# Hidden Layer 2 (16 Neurons)
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

# Hidden Layer 3 (8 Neurons)
model.add(Dense(8, activation='relu'))

# Output Layer (1 Neuron for Price)
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse')

# 4. Early Stopping (The magic trick against Overfitting)
# Stop training if validation loss doesn't improve for 50 epochs, and keep best weights
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# 5. Train Model
print("Training Deep Neural Network (DNN)...")
model.fit(X_train_scaled, y_train_scaled, 
          validation_split=0.2, # Allocate 20% of training data to monitor overfitting in real-time
          epochs=1000, 
          batch_size=8, 
          callbacks=[early_stop], 
          verbose=1)

# 6. Evaluate
y_pred_scaled = model.predict(X_test_scaled)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
mse = mean_squared_error(y_test, y_pred_real)

print("\n--- Final Result (DNN) ---")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):,.0f} Toman")