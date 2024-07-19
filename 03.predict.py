import numpy as np

from lib.model import Model
from lib.data import load_csv

if __name__ == "__main__":
    """
    Program to test neural network model
    """    
    X, y = load_csv("./test.csv")
    
    model = Model().load_model_from_json('trained.json')    

    # Make predictions
    print(f"Accuracy: {model.accuracy(X, y) :.2f}")

    # Evaluate predictions using binary cross-entropy
    loss = model.binary_cross_entropy(y, model.predict(X))
    print(f'Binary Cross-Entropy Loss: {loss}')
