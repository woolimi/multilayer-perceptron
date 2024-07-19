import numpy as np

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu_backward(dA, z):
    dZ = dA.copy()
    dZ[z <= 0] = 0 # Derivative of ReLU
    return dZ

def softmax_backward(dA):
    # Gradient already computed for softmax
    return dA
