import pandas as pd
import numpy as np

# 1. Load data and clean
df = pd.read_csv("./data_scrapper/divar_tara_ml_dataset.csv")
cols_to_drop = ['title', 'token', 'url', 'description_snippet', 'brand_model', 'year']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df = df.dropna(subset=['price_toman'])

# Remove Outliers (IQR)
Q1 = df['price_toman'].quantile(0.25)
Q3 = df['price_toman'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['price_toman'] >= lower_bound) & (df['price_toman'] <= upper_bound)]
df = df.fillna(0)

# Convert text to numbers
cat_cols = ['color', 'body_status', 'engine_status', 'chassis_status', 'gearbox', 'fuel_type', 'airbag', 'tire_quality']
df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)

X_raw = df_encoded.drop(columns=['price_toman']).values.astype(float)
y_raw = df_encoded['price_toman'].values.astype(float).reshape(-1, 1)

# 2. Manual Train/Test Split (80% Train, 20% Test)
np.random.seed(42)
indices = np.random.permutation(len(X_raw))
test_size = int(0.2 * len(X_raw))

train_idx, test_idx = indices[test_size:], indices[:test_size]
X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

# 3. Normalization (Min-Max Scaling) - CRITICAL FOR NEURAL NETWORKS
X_min, X_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
X_max = np.where(X_max == X_min, X_max + 1, X_max) # Prevent division by zero
X_train = (X_train_raw - X_min) / (X_max - X_min)
X_test = (X_test_raw - X_min) / (X_max - X_min)

y_min, y_max = y_train_raw.min(), y_train_raw.max()
y_train = (y_train_raw - y_min) / (y_max - y_min)

# 4. Neural Network Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sig_deriv(x):
    return x * (1 - x)

# 5. Network Architecture (2 Hidden Layers)
input_size = X_train.shape[1]
hidden1_size = 8
hidden2_size = 4
hidden3_size = 4
output_size = 1

# Init weights randomly
W1 = np.random.randn(input_size, hidden1_size) * 0.1
W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
W3 = np.random.randn(hidden2_size, hidden3_size) * 0.1
W4 = np.random.randn(hidden3_size, output_size) * 0.1

epochs = 15000
learning_rate = 0.05

print("Training 2-Hidden-Layer Neural Network...")
for epoch in range(epochs):
    # --- Forward Pass (Make a prediction) ---
    layer1 = sigmoid(np.dot(X_train, W1))
    layer2 = sigmoid(np.dot(layer1, W2))
    layer3 = sigmoid(np.dot(layer2, W3))
    output = sigmoid(np.dot(layer3, W4))
    
    # --- Calculate Error ---
    error = output - y_train
    
    # --- Backpropagation (Learn from mistakes) ---
    d_output = error * sig_deriv(output)
    
    error_layer3 = d_output.dot(W4.T)
    d_layer3 = error_layer3 * sig_deriv(layer3)
    
    error_layer2 = d_layer3.dot(W3.T)
    d_layer2 = error_layer2 * sig_deriv(layer2)
    
    error_layer1 = d_layer2.dot(W2.T)
    d_layer1 = error_layer1 * sig_deriv(layer1)
    
    W4 -= layer3.T.dot(d_output) * learning_rate
    W3 -= layer2.T.dot(d_layer3) * learning_rate
    W2 -= layer1.T.dot(d_layer2) * learning_rate
    W1 -= X_train.T.dot(d_layer1) * learning_rate
    
    if epoch % 3000 == 0:
        print(f"Epoch {epoch} | Scaled Loss: {np.mean(error ** 2):.5f}")

# 6. Evaluate on Test Data
test_layer1 = sigmoid(np.dot(X_test, W1))
test_layer2 = sigmoid(np.dot(test_layer1, W2))
test_layer3 = sigmoid(np.dot(test_layer2, W3))
test_output_scaled = sigmoid(np.dot(test_layer3, W4))

# Revert scaling to get real Toman prices
test_pred_real = (test_output_scaled * (y_max - y_min)) + y_min

final_mse = np.mean((y_test_raw - test_pred_real) ** 2)

print("\n--- Final Result (Numpy NNN) ---")
print(f"Mean Squared Error (MSE): {final_mse}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(final_mse):,.0f} Toman")