import sys
sys.path.append("src")

import numpy as np
import pandas as pd

from neural_network import train, predict
'''
uncommented code for loading PCA  HAR dataset and training neural network on it.
train_data = pd.read_csv("data/har_train_pca.csv")
test_data = pd.read_csv("data/har_test_pca.csv")

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values
'''
np.random.seed(42)
# Load original HAR dataset
#commet here before loading PCA HAR dataset and training on it.
X_train = pd.read_csv(
    "data/UCI HAR Dataset/train/X_train.txt",
    sep=r"\s+",
    header=None
).values

X_test = pd.read_csv(
    "data/UCI HAR Dataset/test/X_test.txt",
    sep=r"\s+",
    header=None
).values

y_train = pd.read_csv(
    "data/UCI HAR Dataset/train/y_train.txt",
    header=None
).values.flatten() - 1

y_test = pd.read_csv(
    "data/UCI HAR Dataset/test/y_test.txt",
    header=None
).values.flatten() - 1

#commet unitl here before loading PCA HAR dataset and training on it.

hidden_dim = 64
learning_rate = 0.001
num_iterations = 10000

W1, b1, W2, b2 = train(
    X_train,
    y_train,
    hidden_dim=hidden_dim,
    learning_rate=learning_rate,
    num_iterations=num_iterations
)

preds = predict(X_test, W1, b1, W2, b2)

accuracy = np.mean(preds == y_test)

print("\nTest Accuracy:", accuracy)