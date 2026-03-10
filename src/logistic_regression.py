import pandas as pd
import numpy as np

# The original dataset has 5 classes (0–4) representing disease severity.
# To make multiclass classification more stable with this small dataset,
# we merge classes 2, 3, and 4 into a single category.
# Final labels become: 0 (none), 1 (mild), 2 (moderate/severe)

#data preprocessing steps (uncomment to run)

#path = 'data/heart+disease/processed.cleveland.data' this  is the original data path
#data[13] = data[13].replace({2:2, 3:2, 4:2}) #code to process the data by merging classes 2, 3, and 4 into a single category (2)
#data.to_csv("data/heart_processed.csv", index=False) this is the code to save the processed data to a new csv file
#data = data.replace("?", pd.NA)
#data = data.dropna()
#data = data.astype(float)
#data.to_csv("data/heart_processed.csv", index=False)

path = 'data/heart_processed.csv'  # This is the path to the processed data
data = pd.read_csv(path, header=None)
print(data.head())
#print(data.shape)
#print(data[13].value_counts())
#print(data.isna().sum())
print(data.isna().sum())
print(data.shape)

def calculate_gradient(X, y, theta):

    n, d = X.shape
    K = theta.shape[1]

    gradient = np.zeros_like(theta)

    for i in range(n):

        scores = np.dot(X[i], theta)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        for k in range(K):

            y_k = 1 if y[i] == k else 0
            gradient[:, k] += X[i] * (probs[k] - y_k)

    return gradient / n

def gradient_descent(X, y, theta, learning_rate, num_iterations):

    for _ in range(num_iterations):
        gradient = calculate_gradient(X, y, theta)
        theta -= learning_rate * gradient

    return theta

def softmax_regression(X, y, learning_rate=0.0001, num_iterations=10000):

    n, d = X.shape
    K = len(np.unique(y))

    theta = np.zeros((d, K))

    theta = gradient_descent(X, y, theta, learning_rate, num_iterations)

    return theta

