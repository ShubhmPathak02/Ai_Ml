import numpy as np
import pandas as pd

data = pd.read_csv('fraud_data.csv')
X = data[['Amount']]
y = data['IsFraud']

X = X.values
y = y.values.reshape(-1, 1)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    m = len(y)
    return -1/m * np.sum(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

learning_rate = 0.1
n_iterations = 1000
m = X_b.shape[0]
theta = np.zeros((X_b.shape[1], 1))

for i in range(n_iterations):
    z = X_b.dot(theta)
    y_pred = sigmoid(z)
    error = y_pred - y
    gradients = 1/m * X_b.T.dot(error)
    theta -= learning_rate * gradients

def predict(new_X):
    new_X = (new_X - np.mean(X, axis=0)) / np.std(X, axis=0)
    new_X_b = np.c_[np.ones((new_X.shape[0], 1)), new_X]
    probs = sigmoid(new_X_b.dot(theta))
    return (probs >= 0.5).astype(int)

example = np.array([[1000]])
print(predict(example)[0][0])
