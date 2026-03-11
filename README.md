# Linear vs Nonlinear Predictors from Scratch

This project explores the performance differences between **linear models and shallow neural networks** for classification tasks.

Both models are implemented **entirely from scratch using NumPy**, without relying on machine learning frameworks such as scikit-learn, PyTorch, or TensorFlow.

The goal of the project is to understand the mathematical foundations of machine learning models by implementing the full training pipeline manually, including forward passes, gradient computation, and optimization.

---

# Project Goals

The main objectives of this project are:

- Implement machine learning models **from first principles**
- Compare **linear decision boundaries** vs **nonlinear models**
- Study how model capacity affects classification performance
- Build the full ML workflow including:
  - preprocessing
  - dimensionality reduction
  - training
  - hyperparameter tuning
  - evaluation

---

# Models Implemented

## Softmax Logistic Regression

A linear classifier trained using gradient descent and the softmax cross-entropy loss.

Key characteristics:

- Multiclass classification
- Vectorized gradient computation
- Numerical stability improvements in softmax
- Implemented without external ML libraries

File:

```
src/logistic_regression.py
```

---

## Shallow Neural Network

A one hidden layer neural network with ReLU activation.

Architecture:

```
Input → Linear → ReLU → Linear → Softmax
```

Features:

- Fully vectorized forward pass
- Backpropagation implemented manually
- He initialization for weights
- Cross-entropy loss

File:

```
src/neural_network.py
```

---

# Datasets

## 1. UCI Heart Disease Dataset

Binary classification task predicting whether a patient has heart disease based on clinical measurements.

Example features include:

- age
- resting blood pressure
- cholesterol
- maximum heart rate
- chest pain type

Target:

```
0 = no heart disease  
1 = heart disease
```

Processed dataset stored in:

```
data/heart_processed.csv
```

---

## 2. Human Activity Recognition Dataset (UCI HAR)

A multiclass classification dataset collected from smartphone motion sensors.

The goal is to classify human activities such as:

- walking
- walking upstairs
- walking downstairs
- sitting
- standing
- laying

Dataset characteristics:

- 561 original features
- 7352 training samples
- 2947 test samples
- 6 activity classes

Source:  
UCI Machine Learning Repository — Human Activity Recognition Dataset.

---

# Dimensionality Reduction

The HAR dataset contains **561 features**, which significantly increases training time.

To reduce dimensionality, **Principal Component Analysis (PCA)** was applied.

Results:

- Reduced to **50 principal components**
- Preserved **~87.5% of the original variance**

Processed datasets:

```
data/har_train_pca.csv
data/har_test_pca.csv
```

PCA implementation:

```
data_pre_processing/pca_har_processing.py
```

---

# Training Pipeline

Training scripts are organized by dataset.

## Dataset 1 (Heart Disease)

```
training_dataset1/
```

Contains:

- softmax regression training
- neural network training
- hyperparameter grid search

---

## Dataset 2 (Human Activity Recognition)

```
training_dataset2/
```

Contains:

- neural network training on HAR dataset
- softmax regression training
- hyperparameter search

---

# Hyperparameter Search

Grid search was implemented to explore the effect of:

- hidden layer size
- learning rate
- number of training iterations

Example search space:

```
hidden_dims = [16, 32, 64]
learning_rates = [0.001, 0.003]
iterations = [5000, 10000]
```

---

# Results

## Human Activity Recognition Dataset

Softmax Regression:

```
Test Accuracy: 94.6%
```

Neural Network (1 hidden layer):

```
Test Accuracy: ~90.5%
```

Interestingly, the linear model performed slightly better than the shallow neural network on this dataset.

This suggests the HAR dataset is already highly structured and nearly linearly separable in feature space.

---

# Repository Structure

```
data/
    heart+disease
    UCI HAR Dataset
    har_train_pca.csv
    har_test_pca.csv

data_pre_processing/
    data_processing.py
    pca_har_processing.py

src/
    logistic_regression.py
    neural_network.py

training_dataset1/
    softmax_regression_train.py
    nn_train.py
    grid_search_hyper_param.py

training_dataset2/
    train_softmax_reg.py
    train_nn2.py
    hyper_param_grid_search.py

notebooks/
    analysis.ipynb

plots/

requirements.txt
README.md
```

---

# Requirements

```
numpy
pandas
```

Install with:

```
pip install -r requirements.txt
```

---

# Key Takeaways

- Linear models can perform surprisingly well on structured datasets
- Implementing ML models from scratch helps develop intuition about:
  - gradient descent
  - backpropagation
  - numerical stability
  - model capacity
- Even simple neural networks require careful tuning to outperform linear models