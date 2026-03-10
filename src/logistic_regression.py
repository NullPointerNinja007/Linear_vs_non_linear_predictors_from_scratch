import pandas as pd

# The original dataset has 5 classes (0–4) representing disease severity.
# To make multiclass classification more stable with this small dataset,
# we merge classes 2, 3, and 4 into a single category.
# Final labels become: 0 (none), 1 (mild), 2 (moderate/severe)

#path = 'data/heart+disease/processed.cleveland.data' this  is the original data path
#data[13] = data[13].replace({2:2, 3:2, 4:2}) #code to process the data by merging classes 2, 3, and 4 into a single category (2)
#data.to_csv("data/heart_processed.csv", index=False) this is the code to save the processed data to a new csv file


path = 'data/heart_processed.csv'  # This is the path to the processed data
data = pd.read_csv(path, header=None)
print(data.head())
print(data.shape)
print(data[13].value_counts())

