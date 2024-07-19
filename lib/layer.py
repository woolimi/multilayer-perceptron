import numpy as np
from lib.activation import relu, relu_backward, softmax, softmax_backward, sigmoid, sigmoid_backward

np.random.seed(42)

class InputLayer:
    def __init__(self,
            inputs: np.ndarray, outputs: np.ndarray, 
            inputs_val: np.ndarray, outputs_val: np.ndarray,epochs: int,
            learning_rate: float, batch_size: int = 16, early_stop: bool = False):
        self.type="InputLayer"
        self.n_neurons = inputs.shape[1]
        self.inputs = inputs
        self.outputs = outputs
        self.inputs_val = inputs_val
        self.outputs_val = outputs_val
        self.activation = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stop = early_stop
    
    def count_parameters(self):
        return 0
    
    def to_json(self):
        return {
            "type": self.type,
            "n_neurons": self.n_neurons,            
            "activation": self.activation,
            "batch_size": self.batch_size
        }
    
class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int, activation=None):
        self.type="DenseLayer"
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
    
    def forward(self, inputs, training=True):
        z = np.dot(inputs, self.weights) + self.biases
        a = None
        if self.activation == "softmax":
            a = softmax(z)
        elif self.activation == "sigmoid":
            a = sigmoid(z)
        elif self.activation == "relu" or self.activation is None:
            a = relu(z)

        if training:
            self.inputs = inputs
            self.z = z
            self.a = a
        return a

    
    def backward(self, dA):
        if self.activation == "relu":
            dZ = relu_backward(dA, self.z)
        elif self.activation == "sigmoid":
            dZ = sigmoid_backward(dA, self.z)
        elif self.activation == "softmax":
            dZ = softmax_backward(dA)
        else:
            dZ = dA  # No activation
        dW = np.dot(self.inputs.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.weights.T)
        return dW, db, dA_prev
    
    def count_parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)
    
    def to_json(self):
        return {
            "type": self.type,
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
            "n_neurons": len(self.biases),
            "activation": self.activation
        }