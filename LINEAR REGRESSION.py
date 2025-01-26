import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
def load_dataset():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Please enter the path to your CSV file: ")

    if not os.path.exists(file_path):
        print("Error: File not found. Please check the path and try again.")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    return df

df = load_dataset()

val_size = float(input("Enter the test size (0 < size < 1), enter value less that 0.001 to skip: "))
def val_train_split(df):
    global X_train, X_val, y_train, y_val, val_size
    
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, val_size=val_size, random_state=42)
    return X_train, X_val, y_train, y_val


# Check for zero-variance features
zero_variance_features = df.std() == 0
if zero_variance_features.any():
    print("Warning: The following features have zero variance and will be removed:")
    print(zero_variance_features[zero_variance_features].index.tolist())
    df = df.drop(zero_variance_features[zero_variance_features].index, axis=1)

# Check for missing values
if df.isnull().any().any():
    print("Warning: The dataset contains missing values. Handling them...")
    df = df.fillna(df.mean())  # Fill missing values with the mean of each column

# Check for non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_columns) > 0:
    print("Warning: The following columns are non-numeric and will be removed:")
    print(non_numeric_columns.tolist())
    df = df.drop(non_numeric_columns, axis=1)

target = input("Enter the name of 'TARGET' column: ")
if target not in df.columns:
    print(f"Error: The dataset must contain a {target} column for the labels.")
    sys.exit(1)

# Plot feature distributions
df.hist(bins=20, figsize=(15, 10))
plt.suptitle('Feature Distributions')
plt.show()

X = df.drop(target, axis=1).values
y = df[target].values

epsilon = 1e-8  
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + epsilon)
X = np.hstack((np.ones((X.shape[0], 1)), X))

def mse(y_true, z):
    return np.mean((y_true - z) ** 2)

def gradient_descent(X, y, weights, learning_rate, num_iterations):
    m = len(y)
    with tqdm(total=num_iterations, desc="Training Progress") as pbar:
        for i in range(num_iterations):
            z = np.dot(X, weights)
            y_hat = z
            gradients = np.dot(X.T, (y_hat - y)) / m
            weights -= learning_rate * gradients

            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                print("Warning: Weights contain NaN or infinite values. Stopping training.")
                break

            if i % 100 == 0:
                loss = mse(y, y_hat)
                pbar.set_postfix({"Loss": loss})
                pbar.update(100)
    return weights

np.random.seed()
weights = np.random.rand(X.shape[1])
learning_rate = 0.001
num_iterations = 1000000
weights = gradient_descent(X, y, weights, learning_rate, num_iterations)

def predict(X, weights):
    z = np.dot(X, weights)
    return z

z = predict(X, weights)




def rmse(y_true, z):
    return np.sqrt(mse(y_true, z))

def r2_score(y_true, z):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - z) ** 2)
    return 1 - (ss_residual / ss_total)

def mae(y_true, z):
    return np.mean(np.abs(y_true - z))

# Accuracy metrics
print(f"MSE: {mse(y, z):.4f}")
print(f"RMSE: {rmse(y, z):.4f}")
print(f"R-squared (R²): {r2_score(y, z):.4f}")
print(f"MAE: {mae(y, z):.4f}")

def val(X_val, y_val):
    if val_size> 0.001:
        predictions = predict(X_val, weights)
        print(f"Validation MSE: {mse(y_val, predictions):.4f}")
        print(f"Validation RMSE: {rmse(y_val, predictions):.4f}")
        print(f"Validation R-squared (R²): {r2_score(y_val, predictions):.4f}")
        print(f"Validation MAE: {mae(y_val, predictions):.4f}")
        
    





def predict_user_input():
    print("\n======================")
    print("  User Input Prediction")
    print("======================\n")

    user_data = []
    feature_names = df.drop('target', axis=1).columns

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter value for {feature}  "))
                user_data.append(value)
                break 
            except ValueError:
                print("Invalid input! Please enter a numeric value: ")
    user_data = (np.array(user_data) - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    user_data = np.insert(user_data, 0, 1)  

    prediction = predict(user_data.reshape(1, -1), weights)

    print(f"Predicted target: {int(prediction[0])}")
    
    
def predict_from_csv():
    test_file_path = input("\nEnter the path to your test CSV file: ")

    if not os.path.exists(test_file_path):
        print("Error: File not found. Please check the path and try again.")
        sys.exit(1)

    try:
        test_df = pd.read_csv(test_file_path)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        sys.exit(1)

    # Ensure test dataset has the same feature columns as the training dataset
    if set(df.drop(target, axis=1).columns) != set(test_df.drop(target, axis=1).columns):
        print("Error: Test dataset must have the same feature columns as the training dataset.")
        sys.exit(1)

    # Extract features
    test_X = test_df.drop(target, axis=1).values

    # Ensure test_X is 2D (if it's 1D array, reshape it)
    if test_X.ndim == 1:
        test_X = test_X.reshape(-1, 1)

    # Normalize test features
    test_X = (test_X - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    test_X = np.hstack((np.ones((test_X.shape[0], 1)), test_X)) 

    # Make predictions
    predictions = predict(test_X, weights)

    # Save predictions to CSV
    test_df['Predicted_TARGET'] = predictions
    output_file_path = input("\nEnter the path to save the prediction CSV file (e.g., predictions.csv): ")
    try:
        test_df.to_csv(output_file_path, index=False)
        print(f"Predictions saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)
# Main Program
def main():
    print("\nChoose an option:")
    print("1. Predict using manual input")
    print("2. Predict using a test CSV file")
    choice = input("\nEnter your choice (1 or 2): ")

    if choice == '1':
        predict_user_input()
    elif choice == '2':
        predict_from_csv()
    else:
        print("Invalid choice. Exiting.")    
    

if __name__ == "__main__":
    main()
