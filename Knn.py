import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def manual_input_knn():
    num_classes = int(input("Enter the number of classes: "))
    class_names = []
    data = []
    targets = []

    for i in range(1, num_classes + 1):
        name = input(f"Enter the name for class {i}: ")
        class_names.append(name)

    num_features = int(input("Enter the number of features (dimensions): "))
    print("\nStart entering values for each class. Use '/' to move to the next class.")

    for class_id in range(1, num_classes + 1):
        print(f"\nEntering values for class {class_names[class_id - 1]} (Class {class_id})")
        while True:
            value = input(f"Enter values for class {class_names[class_id - 1]} (comma-separated, or '/' to switch class): ")
            if value == "/":
                break
            try:
                value = value.replace(" ", "")  # Remove all spaces
                features = list(map(float, value.split(',')))  # Convert comma-separated values to list of floats
                if len(features) != num_features:
                    print(f"Please enter {num_features} values for each point.")
                    continue
                data.append(features)
                targets.append(class_id)
            except ValueError:
                print("Invalid input. Please enter numeric values.")

    while True:
        try:
            k = int(input("Enter the value of k for KNN: "))
            if k <= 0:
                print("Please enter a positive integer for k.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer for k.")
    
    while True:
        new_point_input = input(f"Enter {num_features} values (comma-separated) for the new point to predict: ")
        try:
            new_point_input = new_point_input.replace(" ", "")
            new_point = list(map(float, new_point_input.split(',')))
            if len(new_point) != num_features:
                print(f"Please enter exactly {num_features} values.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter numeric values separated by commas.")
    
    data = np.array(data)
    targets = np.array(targets)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    new_point = scaler.transform([new_point])[0]  # Standardize the new point

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data, targets)
    prediction = knn.predict([new_point])

    print(f"Predicted class: {class_names[prediction[0] - 1]}")

    if num_features > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        reduced_new_point = pca.transform([new_point])
        plot_knn_with_pca(reduced_data, targets, reduced_new_point, class_names)
    else:
        plot_knn(data, targets, new_point, class_names)

def plot_knn_with_pca(reduced_data, targets, reduced_new_point, class_names=None):
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    h = 0.02

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(reduced_data, targets)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    unique_classes = np.unique(targets)
    colors = plt.cm.get_cmap('tab10', len(unique_classes))
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFFFAA"]))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=targets, edgecolor='k', cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]))
    plt.scatter(reduced_new_point[0][0], reduced_new_point[0][1], c='yellow', edgecolor='k', label='New Point')
    plt.title("KNN Classification with PCA (Manual Input)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


def plot_knn(data, targets, new_point, class_names=None):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    h = 0.02
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(data, targets)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Draw lines to the nearest neighbor
    ax = plt.subplot()
    ax.grid(True, color='#323232')
    ax.set_facecolor('black')
    ax.figure.set_facecolor('#121212')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF", "#FFFFAA"]))
    plt.scatter(data[:, 0], data[:, 1], c=targets, edgecolor='k', cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]))
    plt.scatter(new_point[0], new_point[1], c='black', edgecolor='k', label='New Point')
    plt.title("KNN Classification (Manual Input)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    [ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color='#104DCA', linestyle='--', linewidth=1) for point in new_point['blue']]
    [ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color='#EF6C35', linestyle='--', linewidth=1) for point in new_point['orange']]
    plt.legend()
    plt.show()

def dataset_input_knn():
    file_path = input("Enter the path to the CSV file: ")
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        print("Dataset Preview:\n", df.head())

        target_column = input("Enter the name of the target column: ")
        if target_column not in df.columns:
            print("Target column not found in the dataset.")
            return

        df = df.dropna(axis=1, how='all')
        df = df.fillna(df.mean())

        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        # Check if the target column is categorical (non-numeric)
        if df[target_column].dtype == 'object':
            # Convert categorical target column to numeric labels (1, 2, etc.)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)  # Convert categorical values to numeric

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        k = int(input("Enter the value of k for KNN: "))
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)

        print("Model trained successfully.")

        print("\nEnter values for prediction (comma-separated for each feature):")
        new_point = list(map(float, input().split(",")))
        new_point = scaler.transform([new_point])[0]  # Standardize the new point
        prediction = knn.predict([new_point])
        print(f"Predicted class: {label_encoder.inverse_transform(prediction)}")

        plot_dataset_knn(X, y, new_point, method="PCA")
        evaluate_model(knn, X, y)

    except Exception as e:
        print("An error occurred:", e)

def plot_dataset_knn(X, y, new_point, method="PCA", class_names=None):
    if X.shape[1] > 2:
        print(f"Dataset has {X.shape[1]} features. Reducing dimensions using {method}.")
        
        if method.upper() == "PCA":
            reducer = PCA(n_components=2)
        elif method.upper() == "TSNE":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            print("Invalid method specified. Using PCA as default.")
            reducer = PCA(n_components=2)
        
        X_reduced = reducer.fit_transform(X)
        new_point_reduced = reducer.transform([new_point]) if method.upper() == "PCA" else reducer.fit_transform(np.vstack([X, new_point]))[-1]

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]), edgecolor='k')
        plt.scatter(new_point_reduced[0], new_point_reduced[1], c='yellow', edgecolor='k', label='New Point')
        plt.title(f"KNN Classification with {method} Reduction")
        plt.xlabel(f"{method} Component 1")
        plt.ylabel(f"{method} Component 2")
        plt.legend()
        plt.colorbar(scatter)
        plt.show()

def evaluate_model(knn, X, y):
    y_pred = knn.predict(X)
    print("\nModel Evaluation:")
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)

    fpr, tpr, _ = roc_curve(y, knn.predict_proba(X)[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.2f}")
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    choice = input("Enter 1 for manual input or 2 for CSV dataset input: ")
    if choice == '1':
        manual_input_knn()
    elif choice == '2':
        dataset_input_knn()
    else:
        print("Invalid choice. Please run the program again and select a valid option.")