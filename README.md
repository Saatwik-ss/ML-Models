# LOGISTIC_PREDICTOR
 
Logistic Regression and Data Visualization for Binary Classification

This repository provides an implementation of a logistic regression model from scratch, along with data visualization and evaluation techniques for binary classification. Here’s a breakdown of the project:

1. Data Loading and Visualization
Feature Distribution: Loads a CSV dataset and visualizes feature distributions with histograms to explore feature variability.
Correlation Heatmap: Displays correlations between features to identify potential relationships.
2. Logistic Regression Model Implementation
Data Preprocessing: Normalizes features and adds an intercept term for logistic regression.
Model Functions:
sigmoid: Calculates the sigmoid function for logistic regression.
compute_loss: Computes binary cross-entropy loss.
gradient_descent: Updates model weights using gradient descent, printing the loss every 1,000 iterations.
3. Model Evaluation
Predictions and Metrics:
accuracy: Overall prediction accuracy.
precision: Proportion of true positives out of all positive predictions.
recall: Proportion of true positives out of all actual positives.
f1_score: Harmonic mean of precision and recall.
4. Feature Importance
Feature Coefficients: Visualizes model coefficients to show each feature’s impact on predictions.
5. Confusion Matrix
Confusion Matrix Visualization: Displays a matrix to show counts of true positives, true negatives, false positives, and false negatives.
6. ROC Curve
Receiver Operating Characteristic (ROC) Curve: Plots the ROC curve to assess the model's true positive and false positive rates across thresholds.
