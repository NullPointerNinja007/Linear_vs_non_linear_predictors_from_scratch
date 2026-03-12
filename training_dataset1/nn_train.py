import sys
sys.path.append("src")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from neural_network import train, predict

np.random.seed(7)

# Load dataset
data = pd.read_csv("data/heart_processed.csv", header=None)
X = data.drop(13, axis=1).values
y = data[13].values.astype(int)


print("Class counts:")
print(pd.Series(y).value_counts())


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train neural network
    W1, b1, W2, b2 = train(
        X_train,
        y_train,
        hidden_dim=8,
        learning_rate=0.001,
        num_iterations=10000
    )

    # Predictions
    preds = predict(X_test, W1, b1, W2, b2)

    acc = np.mean(preds == y_test)
    accuracies.append(acc)


print("Fold accuracies:", accuracies)
print("Mean accuracy:", np.mean(accuracies))
print("Std accuracy:", np.std(accuracies))