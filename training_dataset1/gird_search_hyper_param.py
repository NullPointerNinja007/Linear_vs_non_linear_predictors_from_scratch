"""Grid search for hyperparameter tuning (this is not a true grid search, but a manual loop over combinations)
for our neural network implementation. We will test different combinations of
hidden layer sizes, learning rates, and number of iterations to find the best
configuration for our dataset."""

import sys
sys.path.append("src")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from neural_network import train, predict


# Load dataset
data = pd.read_csv("data/heart_processed.csv", header=None)

X = data.drop(13, axis=1).values
y = data[13].values.astype(int)

# Hyperparameter grid
hidden_dims = [4, 8, 16, 32]
learning_rates = [0.001, 0.005]
iterations_list = [5000, 10000, 20000]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for hidden_dim in hidden_dims:
    for lr in learning_rates:
        for iters in iterations_list:

            accuracies = []

            for train_index, test_index in skf.split(X, y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                W1, b1, W2, b2 = train(
                    X_train,
                    y_train,
                    hidden_dim=hidden_dim,
                    learning_rate=lr,
                    num_iterations=iters
                )

                preds = predict(X_test, W1, b1, W2, b2)
                acc = np.mean(preds == y_test)

                accuracies.append(acc)

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            results.append((hidden_dim, lr, iters, mean_acc, std_acc))

            print(
                f"hidden={hidden_dim}, lr={lr}, iters={iters} "
                f"-> mean={mean_acc:.4f}, std={std_acc:.4f}"
            )


print("\nBest configuration:")
best = max(results, key=lambda x: x[3])
print(best)