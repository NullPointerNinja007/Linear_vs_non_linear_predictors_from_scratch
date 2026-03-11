import pandas as pd
import numpy as np



def calculate_gradient(X, y, theta):

    n = X.shape[0]
    K = theta.shape[1]

    # Compute scores for all samples
    scores = X @ theta

    # Numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)

    # Softmax probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # One-hot encode labels
    Y = np.zeros_like(probs)
    Y[np.arange(n), y] = 1

    # Gradient
    gradient = X.T @ (probs - Y) / n

    return gradient

def gradient_descent(X, y, theta, learning_rate, num_iterations):

    for _ in range(num_iterations):
        gradient = calculate_gradient(X, y, theta)
        theta -= learning_rate * gradient

    return theta

def softmax_regression(X, y, learning_rate=0.0001, num_iterations=10000):

    X = np.column_stack((np.ones(X.shape[0]), X))
    n, d = X.shape
    K = len(np.unique(y))

    theta = np.zeros((d, K))

    theta = gradient_descent(X, y, theta, learning_rate, num_iterations)

    return theta

def predict(X, theta):
    """
    Predict class labels using trained softmax regression parameters.
    """
    X = np.column_stack((np.ones(X.shape[0]), X))

    scores = np.dot(X, theta)

    # numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)