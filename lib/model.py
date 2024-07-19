import matplotlib.pyplot as plt
import numpy as np
import json
from lib.layer import InputLayer, DenseLayer
from lib.print import warning

class Model:
    def __init__(self):
        self.layers = []
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracy = []
        self.validation_accuracy = []
        self.training_mses = []
        self.validation_mses = []
        self.inputLayer = None       
        self.train_patience = 10

    def create_network(self, layers):
        if not isinstance(layers[0], InputLayer):
            raise Exception("First layer must be an InputLayer")
        self.inputLayer = layers[0]
        self.layers = layers[1:]
        return self
    
    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output
        
    def backward(self, y_true: np.ndarray):
        m = y_true.shape[0]
        # Result of last layer (softmax result)
        y_pred = self.layers[-1].a
        dA = y_pred.copy()
        dA[range(m), y_true] -= 1
        dA /= m
        
        for layer in reversed(self.layers):
            dW, db, dA = layer.backward(dA)
            layer.dW = dW
            layer.db = db
    
    # w_new = w_old - learning_rate * d_J / d_w
    # b_new = b_old - learning_rate * d_J / d_b
    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dW
            layer.biases -= learning_rate * layer.db
    
    def plot_loss(self):
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.legend()
        plt.savefig('plot-loss.png')
        plt.close()
    
    def plot_accuracy(self):
        plt.plot(self.training_accuracy, label='Training Accuracy')
        plt.plot(self.validation_accuracy, label='Validation Accuracy')
        plt.legend()
        plt.savefig('plot-accuracy.png')
        plt.close()
    
    def plot_mse(self):
        plt.plot(self.training_mses, label='Training MSE')
        plt.plot(self.validation_mses, label='Validation MSE')
        plt.legend()
        plt.savefig('plot-mse.png')
        plt.close()

    def get_losses(self, y_batch_pred, y_batch_true, y_val_pred, y_val):
        training_loss = self.binary_cross_entropy(y_batch_true, y_batch_pred)
        validation_loss = self.binary_cross_entropy(y_val, y_val_pred)
        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)
        return training_loss, validation_loss
    
    def get_accuracy(self, X_batch, y_batch_true, X_val, y_val):
        training_accuracy = self.accuracy(X_batch, y_batch_true)
        validation_accuracy = self.accuracy(X_val, y_val)
        self.training_accuracy.append(training_accuracy)
        self.validation_accuracy.append(validation_accuracy)
        return training_accuracy, validation_accuracy

    # Mean Squared Error
    def get_mse(self, y_batch_pred, y_batch_true, y_val_pred, y_val):
        training_mse = self.mse(y_batch_true, y_batch_pred)
        validation_mse = self.mse(y_val, y_val_pred)
        self.training_mses.append(training_mse)
        self.validation_mses.append(validation_mse)
        return training_mse, validation_mse

    def train(self):
        X = self.inputLayer.inputs
        y = self.inputLayer.outputs
        X_val = self.inputLayer.inputs_val
        y_val = self.inputLayer.outputs_val
        batch_size = self.inputLayer.batch_size
        epochs = self.inputLayer.epochs
        learning_rate = self.inputLayer.learning_rate
        early_stop = self.inputLayer.early_stop
        
        for epoch in range(1, epochs + 1):
            batch_idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
            X_batch = X[batch_idx]
            y_batch_true = y[batch_idx]
    
            y_batch_pred = self.forward(X_batch)
            y_val_pred = self.predict(X_val)

            # Calculate Loss
            validation_loss, training_loss = self.get_losses(y_batch_pred, y_batch_true, y_val_pred, y_val)

            # Calculate Accuracy
            validation_accuracy, training_accuracy = self.get_accuracy(X_batch, y_batch_true, X_val, y_val)

            # Calculate Mean Squared Error
            training_mse, validation_mse = self.get_mse(y_batch_pred, y_batch_true, y_val_pred, y_val)

            # Early Stopping
            if early_stop:
                if len(self.validation_losses) > 2 and validation_loss > self.validation_losses[-2]:
                    self.train_patience -= 1
                if self.train_patience == 0:
                    print(f"epoch {epoch:>4}/{epochs:>4} - Training stopped due to early stopping")
                    break

            # Update model
            self.backward(y_batch_true)
            self.update_weights(learning_rate)
            if epoch % 100 == 0:
                print(f'epoch {epoch:>4}/{epochs:>4} - loss: {training_loss:>6.6f} val_loss: {validation_loss:>6.6f} acc: {training_accuracy:>3.0f}%, val_acc: {validation_accuracy:>3.0f}%, mse: {training_mse:>6.6f}, val_mse: {validation_mse:>6.6f}')
        
        self.plot_loss()
        self.plot_accuracy()
        self.plot_mse()
    
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
                self.inputLayer = InputLayer(np.array([[]]), np.array([[]]), np.array([[]]), np.array([[]]), 0, 0, 0)
            if (layer_data["type"] == "DenseLayer"):
                denseLayer = DenseLayer(len(layer_data["biases"]), layer_data["n_neurons"], layer_data["activation"])
                denseLayer.weights = np.array(layer_data["weights"])
                denseLayer.biases = np.array(layer_data["biases"])
                self.layers.append(denseLayer)
        return self

    def predict(self, X):
        return self.forward(X, training=False)

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return np.sum(y_true == y_pred.argmax(axis=1)) / len(y_true) * 100
    
    def mse(self, y_true, y_pred):
        y_pred = y_pred[:, 1] # Extract probability of class 1
        return np.mean((y_true - y_pred)**2)
    
    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = y_pred[:, 1] # Extract probability of class 1
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
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
