import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def calculate_loss(P, Y):
    n = Y.shape[0]
    log_likelihood = -np.log(P[range(n), Y] + 1e-15)
    loss = np.sum(log_likelihood) / n
    return loss

def forward_pass(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    A1 = relu(z1)
    z2 = np.dot(A1, W2) + b2
    P = softmax(z2)
    return z1, A1, z2, P

def backward_pass(X, Y, z1, A1, P, W2):
    n = X.shape[0]
    d2 = P.copy()
    d2[range(n), Y] -= 1
    d2 /= n

    dW2 = np.dot(A1.T, d2)
    db2 = np.sum(d2, axis=0, keepdims=True)

    dA1 = np.dot(d2, W2.T)
    dZ1 = dA1 * relu_derivative(z1)

    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

def initialize_parameters(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, _, _, P = forward_pass(X, W1, b1, W2, b2)
    return np.argmax(P, axis=1)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def train(X, Y, hidden_dim=32, learning_rate=0.01, num_iterations=10000):
    input_dim = X.shape[1]
    output_dim = len(np.unique(Y))

    W1, b1, W2, b2 = initialize_parameters(input_dim, hidden_dim, output_dim)

    for i in range(num_iterations):
        z1, A1, z2, P = forward_pass(X, W1, b1, W2, b2)
        loss = calculate_loss(P, Y)

        dW1, db1, dW2, db2 = backward_pass(X, Y, z1, A1, P, W2)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return W1, b1, W2, b2