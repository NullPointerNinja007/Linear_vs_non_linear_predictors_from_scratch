import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from logistic_regression import softmax_regression, predict

# Load data
data = pd.read_csv("data/heart_processed.csv", header=None)

X = data.drop(13, axis=1).values
y = data[13].values.astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Class counts:")

accuracies = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit scaler only on training fold
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    theta = softmax_regression(
        X_train,
        y_train,
        learning_rate=0.01,
        num_iterations=5000
    )

    # Evaluate
    preds = predict(X_test, theta)
    acc = np.mean(preds == y_test)
    accuracies.append(acc)

print("Fold accuracies:", accuracies)
print("Mean accuracy:", np.mean(accuracies))
print("Std accuracy:", np.std(accuracies))