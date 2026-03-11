import sys
sys.path.append("src")

import numpy as np
import pandas as pd

from neural_network import train, predict

# Load datasets
train_data = pd.read_csv("data/har_train_pca.csv")
test_data = pd.read_csv("data/har_test_pca.csv")

X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Hyperparameter grid
hidden_dims = [16, 32, 64]
learning_rates = [0.001, 0.003]
iterations_list = [5000, 10000]

results = []

for hidden_dim in hidden_dims:
    for lr in learning_rates:
        for iters in iterations_list:

            print("\n--------------------------------")
            print(f"hidden_dim={hidden_dim}, lr={lr}, iters={iters}")

            W1, b1, W2, b2 = train(
                X_train,
                y_train,
                hidden_dim=hidden_dim,
                learning_rate=lr,
                num_iterations=iters
            )

            preds = predict(X_test, W1, b1, W2, b2)
            acc = np.mean(preds == y_test)

            print("Test accuracy:", acc)

            results.append((hidden_dim, lr, iters, acc))


# Print results
print("\nAll results:")
for r in results:
    print(r)

best = max(results, key=lambda x: x[3])

print("\nBest configuration:")
print(best)