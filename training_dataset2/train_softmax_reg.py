import sys
sys.path.append("src")

import numpy as np
import pandas as pd

from logistic_regression import softmax_regression, predict

np.random.seed(42)
# Load datasets
'''
train_data = pd.read_csv("data/har_train_pca.csv")
test_data = pd.read_csv("data/har_test_pca.csv")

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Split features and labels
X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values.astype(int)

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values.astype(int)
'''
# Load original HAR dataset
#comment from here before loading PCA HAR dataset and training on it.
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

#comment until here before loading PCA HAR dataset and training on it.

# Train softmax regression
theta = softmax_regression(
    X_train,
    y_train,
    learning_rate=0.01,
    num_iterations=10000
)

# Predict
preds = predict(X_test, theta)

# Accuracy
accuracy = np.mean(preds == y_test)

print("\nSoftmax Regression Test Accuracy:", accuracy)

# Optional: show class counts
print("\nClass distribution in training data:")
print(pd.Series(y_train).value_counts())