import numpy as np
import json
from lib.layer import InputLayer, DenseLayer

class Model:
    def __init__(self):
        self.layers = []
        self.inputLayer = None        

    def create_network(self, layers):
        if not isinstance(layers[0], InputLayer):
            raise Exception("First layer must be an InputLayer")
        self.inputLayer = layers[0]
        self.layers = layers[1:]
        return self
    
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, X, y_true):
        m = y_true.shape[0]
        y_pred = self.layers[-1].a
        dA = y_pred
        dA[range(m), y_true] -= 1
        dA /= m
        
        for layer in reversed(self.layers):
            dW, db, dA = layer.backward(dA)
            layer.dW = dW
            layer.db = db
    
    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dW
            layer.biases -= learning_rate * layer.db
    
    def train(self):
        X = self.inputLayer.inputs
        y = self.inputLayer.outputs
        batch_size = self.inputLayer.batch_size
        epochs = self.inputLayer.epochs
        learning_rate = self.inputLayer.learning_rate

        for epoch in range(epochs):
            batch_idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
    
            y_pred = self.forward(X_batch)
            loss = self.compute_loss(y_batch, y_pred)
            self.backward(X_batch, y_batch)
            self.update_weights(learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def save_model_as_json(self, filename):
        model_data = []
        for layer in [self.inputLayer] + self.layers:
            model_data.append(layer.to_json())
        with open(filename, 'w') as f:
            json.dump(model_data, f)
    
    def load_model_from_json(self, filename):
        with open(filename, 'r') as f:
            model_data = json.load(f)
        for layer_data in model_data:
            if (layer_data["type"] == "InputLayer"):
                self.inputLayer = InputLayer(np.array([[]]), np.array([[]]), 0, 0, 0)
            if (layer_data["type"] == "DenseLayer"):
                denseLayer = DenseLayer(len(layer_data["biases"]), layer_data["n_neurons"], layer_data["activation"])
                denseLayer.weights = np.array(layer_data["weights"])
                denseLayer.biases = np.array(layer_data["biases"])
                self.layers.append(denseLayer)
        return self

    def predict(self, X):
        return self.forward(X).argmax(axis=1)
    
    def binary_cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def summary(self):
        print(f"{'Layer (type)': <18} {'Output Shape': <16} {'Param #':<10} {'Activation'}")
        print("="*60)
        total_params = 0

        for layer in [self.inputLayer] + self.layers:
            layer_type = layer.type
            param_count = layer.count_parameters()
            output_shape = layer.n_neurons
            activation = layer.activation
            total_params += param_count
            print(f"{layer_type: <18} {str(output_shape): <16} {param_count: <10} {activation}")
        print("="*60)
        print(f"Total parameters: {total_params}")