import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load HAR data
X_train = pd.read_csv("data/UCI HAR Dataset/train/X_train.txt", sep="\s+", header=None)
y_train = pd.read_csv("data/UCI HAR Dataset/train/y_train.txt", header=None)

X_test = pd.read_csv("data/UCI HAR Dataset/test/X_test.txt", sep="\s+", header=None)
y_test = pd.read_csv("data/UCI HAR Dataset/test/y_test.txt", header=None)

# Convert labels to 0–5 instead of 1–6
y_train = y_train.values.flatten() - 1
y_test = y_test.values.flatten() - 1

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Explained variance:", pca.explained_variance_ratio_.sum())

# Combine features + labels
train_data = pd.DataFrame(X_train_pca)
train_data["label"] = y_train

test_data = pd.DataFrame(X_test_pca)
test_data["label"] = y_test

# Save
train_data.to_csv("data/har_train_pca.csv", index=False)
test_data.to_csv("data/har_test_pca.csv", index=False)

print("Saved PCA datasets:")
print("data/har_train_pca.csv")
print("data/har_test_pca.csv")