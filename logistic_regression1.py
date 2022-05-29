import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import matplotlib.pyplot as plt

# prepare the data
from sklearn import datasets

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]
def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test
def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

data = datasets.load_iris()
X = normalize(data.data[data.target != 0])
y = data.target[data.target != 0]
y[y == 1] = 0
y[y == 2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)
import math
w, b = (np.zeros((X_train.shape[1], 1)), 0)
n_features = X_train.shape[1]
limit = 1 / math.sqrt(n_features)

w = np.random.uniform(-limit, limit, (n_features,1))
print (w.shape)
print (b)

learning_rate = 0.1
n_epochs = 4000

def propagate(X, y, w, b):
    Y = y.reshape(-1, 1)
    m = Y.shape[0]
    A = sigmoid(np.dot(X, w) + b)
    # A = A.reshape(-1)
    cost = -np.sum((Y*np.log(A)+(1-Y)*np.log(1-A)))/m                                  # compute cost
    err = A - Y
    dw = np.dot(X.T, err)/m
    db = np.sum(err)/m
    return dw, db, cost

for epoch in range(n_epochs):
    import pdb;pdb.set_trace()
    dw, db, cost = propagate(X_train, y_train, w, b)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    print (f"For epoch {epoch} Cost: {round(cost, 3)}")


# predict func
def predict(w, b, X):
    m = X.shape[0]
    A = sigmoid(np.dot(X, w) + b)
    Y_prediction = np.zeros((1,m))
    for i in range(A.shape[1]):
        if(A[0][i] <= 0.5):
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
    return Y_prediction


Y_prediction_test = predict(w, b, X_test)
Y_prediction_train = predict(w, b, X_train)


# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
