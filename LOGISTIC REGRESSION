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


val_size = float(input("Enter the test size (0 < size < 1), enter value less that 0.001 to skip: "))
def val_train_split(df):
    global X_train, X_val, y_train, y_val, val_size
    
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, val_size=val_size, random_state=42)
    return X_train, X_val, y_train, y_val


# Plot feature distributions
df.hist(bins=20, figsize=(15, 10))
plt.suptitle('Feature Distributions')
plt.show()

def correlationheatmap(data):
    correlation = data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(correlation, cmap='coolwarm')
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    fig.colorbar(cax)
    plt.title('Correlation Heatmap', pad=20)
    plt.show()

correlationheatmap(df)


X = df.drop(target, axis=1).values
y = df[target].values

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept term

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))  # Added small value to avoid log(0)

def gradient_descent(X, y, weights, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        z = np.dot(X, weights)
        y_hat = sigmoid(z)
        gradients = np.dot(X.T, (y_hat - y)) / m
        weights -= learning_rate * gradients
        if i % 1000 == 0:
            loss = compute_loss(y, y_hat)
            print(f"Iteration {i}, Loss: {loss:.4f}")
    return weights

np.random.seed()
weights = np.random.rand(X.shape[1])
learning_rate = 0.001
num_iterations = 10000
weights = gradient_descent(X, y, weights, learning_rate, num_iterations)

def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z) >= 0.5

y_pred = predict(X, weights)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0

def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

# Print evaluation metrics
print(f"Accuracy: {accuracy(y, y_pred):.4f}")
print(f"Precision: {precision(y, y_pred):.4f}")
print(f"Recall: {recall(y, y_pred):.4f}")
print(f"F1-Score: {f1_score(y, y_pred):.4f}")

feature_names = ['Intercept'] + list(df.drop(target, axis=1).columns)
coefficients = weights
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients, color='blue')
plt.title("Feature Coefficients (Effect on Prediction)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2))
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # False positives
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # True positives

    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, int(value), ha='center', va='center', color='black')

    plt.show()

plot_confusion_matrix(y, y_pred)

def plot_roc_curve(X, y, weights):
    z = np.dot(X, weights)
    y_hat = sigmoid(z)
    fpr, tpr = [], []

    for threshold in np.linspace(0, 1, 100):
        y_pred_threshold = y_hat >= threshold
        true_positives = np.sum((y == 1) & (y_pred_threshold == 1))
        false_positives = np.sum((y == 0) & (y_pred_threshold == 1))
        actual_positives = np.sum(y == 1)
        actual_negatives = np.sum(y == 0)

        tpr.append(true_positives / actual_positives if actual_positives != 0 else 0)
        fpr.append(false_positives / actual_negatives if actual_negatives != 0 else 0)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="Logistic Regression")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

plot_roc_curve(X, y, weights)

def plot_weighted_contributions(X, y, weights):
    z = np.dot(X, weights) 
    sorted_indices = np.argsort(z)  
    z_sorted = z[sorted_indices]
    y_sorted = y[sorted_indices]
    sigmoid_curve = sigmoid(z_sorted)
    plt.figure(figsize=(12, 6))
    plt.scatter(z, y, label='Data Points', color='blue', alpha=0.7)
    plt.plot(z_sorted, sigmoid_curve, label='Sigmoid Curve', color='red', linewidth=2)
    plt.axhline(y.mean(), color='green', linestyle='--', label='Mean Target')
    plt.title(f"Log-Odds (Sum of Contributions) vs {target}")
    plt.xlabel("Log-Odds (X * weights)")
    plt.ylabel(target)
    plt.legend()
    plt.show()
    
plot_weighted_contributions(X, y, weights)


def val(X_val, y_val):
    if val_size> 0.001:
        predictions = predict(X_val, weights)
        print(f"Validation Accuracy: {accuracy(y_val, predictions):.4f}")
        print(f"Validation Precision: {precision(y_val, predictions):.4f}")
        print(f"Validation Recall: {recall(y_val, predictions):.4f}")
        print(f"Validation F1-Score: {f1_score(y_val, predictions):.4f}")


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
    print(f"NOTE: THE TRAINING DATA PREDICTION HAD AN ACCURACY OF {accuracy(y, y_pred) * 100:.2f}%")
    
    
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

    # Add intercept term
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
