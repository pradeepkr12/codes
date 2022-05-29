import numpy as np
import matplotlib.pyplot as plt
def SSE(y, y_hat):
   return 0.50*np.sum((y - y_hat)**2)
def Linear(Z):
   return Z
def dEdW_1 (y, y_hat,x_1):
   return np.sum(-(y - y_hat) * x_1)
def dEdW_0 (y, y_hat):
   return np.sum(-(y - y_hat))
N = 200
slope = 3
mu, sigma = 0, 0.1 # mean and standard deviation
intercept = np.random.normal(mu, sigma, N)
x_1 = np.random.rand(N)
y = slope*x_1 + intercept
area = np.pi*10
w_1 = np.random.uniform(-2,-3,1)
w_0 = np.random.uniform(0,2,1)
x_0 = 1
y_hat = w_1*x_1 + w_0*x_0
epoch = 300
eta = 0.001
E = []
for ep in range(epoch):
    Z = w_1 * x_1 + w_0*x_0
    y_hat = Linear(Z)
    error = SSE(y, y_hat)
    E.append(error)
    dEdw_1 = dEdW_1(y, y_hat, x_1)
    dEdw_0 = dEdW_0(y, y_hat)
    w_1 = w_1 - eta*dEdw_1
    w_0 = w_0 - eta*dEdw_0
    if ep % 20 == 0:
       plt.scatter(x_1, y, s=area, c='r', alpha=0.5)
       plt.plot(x_1, y_hat, '-b', label="SSE= %.2f" %error + '\n' + "Epoch Number= %d" %(ep))
       plt.xlabel('x (Our Data)')
       plt.ylabel('y_hat (Network Outputs)')
       plt.legend()
       plt.ylim(-3, 5)
       plt.grid()
       plt.show()
plt.plot(E, 'r')
plt.grid()
plt.title("Error")
plt.xlabel("Epochs")
plt.ylabel("SSE")
plt.show()
