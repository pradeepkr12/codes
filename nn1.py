import numpy as np


# create a sample data
x = np.random.randn(100)
y = 2 * x + 0.8

n_epochs=100

w = np.random.randn(1)
b = np.random.randn(1)

learning_rate = 0.001

for _ in range(n_epochs):
    # forward prop
    y_pred = w * x + b
    err = y_pred - y
    w = w - learning_rate * np.sum( x * err)
    b = b - learning_rate * np.sum(err)
    print (w, b)
