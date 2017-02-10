# Introduction to Github!
# A simple python implementation of two layer neural network.

import numpy as np
import matplotlib.pyplot as plt

# 2 Layer Neural Network:
# sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

dataX = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [1]])
b = np.array([[1], [1], [1], [1]])
X = np.concatenate((dataX, b), axis=1)
W1 = 2*np.random.random((3,1))-1
errors = []

for i in range(60000):
    
    # forward propagation through layers 0[input:4x3], and 1[output:3x1]
    dot1 = np.dot(X, W1)
    h1 = sigmoid(dot1)
    
    # backward propagation; error calculation
    # Y = ground truth; h2 = estemated truth
    
    error1 = Y - h1
    del_h1 = sigmoid_derivative(h1) * error1
    
    #updating parameters
    W1 += np.dot(X.T, del_h1)
    
    if i % 1000 == 0:
        #print('Error :', np.mean(np.mean(np.abs(error1))))
        errors.append(np.mean(np.mean(np.abs(error1))))
        
# predicting here
print('predicting here')
dot1 = np.dot(X, W1)
h1 = sigmoid(dot1)
print(h1)
plt.plot(errors)
plt.show()
