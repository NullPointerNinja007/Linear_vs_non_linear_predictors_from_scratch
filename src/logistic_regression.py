import pandas as pd
import numpy as np



def calculate_gradient(X, y, theta):

    n, d = X.shape
    K = theta.shape[1]

    gradient = np.zeros_like(theta)

    for i in range(n):

        scores = np.dot(X[i], theta)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        for k in range(K):

            y_k = 1 if y[i] == k else 0
            gradient[:, k] += X[i] * (probs[k] - y_k)

    return gradient / n

def gradient_descent(X, y, theta, learning_rate, num_iterations):

    for _ in range(num_iterations):
        gradient = calculate_gradient(X, y, theta)
        theta -= learning_rate * gradient

    return theta

def softmax_regression(X, y, learning_rate=0.0001, num_iterations=10000):

    n, d = X.shape
    K = len(np.unique(y))

    theta = np.zeros((d, K))

    theta = gradient_descent(X, y, theta, learning_rate, num_iterations)

    return theta

