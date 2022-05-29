import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# preparing the data
N = 2000
true_X = np.random.normal(0, 1, N)
true_X = true_X.reshape(-1, 1)
true_w = np.asarray(2)
true_b = np.asarray(4)


true_y = np.dot(true_X, true_w) + true_b

# creating a data iterator

def data_iter(features, labels, batch_size=100):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.asarray(indices[i:min(i +
                                                   batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


n_epochs = 10
w = np.asarray(0)
b = np.asarray(0)
learning_rate = 0.001

def squared_loss(y_hat, y):
    """Squared loss."""
    return np.sum((y_hat - y.reshape(y_hat.shape))**2 / 2) / y.shape[0]

step = 0
for epoch in range(n_epochs):
    for X, y in data_iter(true_X, true_y, batch_size=50):
        y_pred = np.dot(X, w) + b
        loss = squared_loss(y_pred, y)
        # update the weights
        w = w - learning_rate * np.dot(X.T, (y_pred - y)) / X.shape[0]
        b = b - learning_rate * np.sum(y_pred - y) / X.shape[0]
        print (f"Loss at epoch: {epoch}, step: {step} is {loss}")
        step += 1
