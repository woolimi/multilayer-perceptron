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
    y_pred = model.predict(X)
    print(f"Accuracy: {np.sum(y == y_pred) / len(y) * 100 :.2f}")

    # Evaluate predictions using binary cross-entropy
    # loss = model.binary_cross_entropy(y, y_pred)
    # print(f'Binary Cross-Entropy Loss: {loss}')
