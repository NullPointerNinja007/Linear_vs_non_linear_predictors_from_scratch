'this is file to run tests'

import pandas as pd


train_data = pd.read_csv("data/har_train_pca.csv")

test_data = pd.read_csv("data/har_test_pca.csv")

print('Train data shape:', train_data.shape)
print('Test data shape:', test_data.shape)
print(train_data.head())
print(train_data["label"].value_counts())