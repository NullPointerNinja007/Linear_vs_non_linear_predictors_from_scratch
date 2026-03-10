import pandas as pd

"""
Data preprocessing script for the UCI Cleveland Heart Disease dataset.

Steps performed:
1. Load the original dataset
2. Replace missing values ("?") with NaN
3. Remove rows containing missing values
4. Convert all values to numeric
5. Merge disease classes (2, 3, 4) into a single class (2)
6. Save the cleaned dataset as heart_processed.csv
"""

# Path to the original dataset
RAW_PATH = "data/heart+disease/processed.cleveland.data"

# Path where the cleaned dataset will be saved
OUTPUT_PATH = "data/heart_processed.csv"


def preprocess_data():
    
    # Load raw dataset
    data = pd.read_csv(RAW_PATH, header=None)

    # Replace missing values marked as "?"
    data = data.replace("?", pd.NA)

    # Remove rows with missing values
    data = data.dropna()

    # Convert all columns to numeric
    data = data.astype(float)

    # Merge classes: 2,3,4 -> 2
    data[13] = data[13].replace({2: 2, 3: 2, 4: 2})

    # Save cleaned dataset
    data.to_csv(OUTPUT_PATH, index=False)

    # Print dataset info
    print("Cleaned dataset saved to:", OUTPUT_PATH)
    print("Dataset shape:", data.shape)
    print("Class distribution:")
    print(data[13].value_counts().sort_index())


if __name__ == "__main__":
    preprocess_data()