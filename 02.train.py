from lib.print import warning
from lib.model import Model
from lib.layer import InputLayer, DenseLayer
from lib.data import load_csv, data_processing


if __name__ == "__main__":
    """
    Program to train artificial neural network model
    """

    X, y = data_processing(load_csv("./data_training.csv"))
    X_val, y_val = data_processing(load_csv("./data_test.csv"))

    print(warning("Loading dataset..."))
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    print(warning("\nCreating neural network model..."))
    # ReLU:
    model = Model().create_network([
        InputLayer(X, y, X_val, y_val, epochs=30000, learning_rate=0.01, batch_size=X.shape[0], early_stop=True),
        DenseLayer(n_inputs=X.shape[1], n_neurons=20,  activation="relu"),
        DenseLayer(n_inputs=20, n_neurons=30, activation="relu"),
        DenseLayer(n_inputs=30, n_neurons=2, activation="softmax"),
    ])
    # Sigmoid
    # model = Model().create_network([
    #     InputLayer(X, y, X_val, y_val, epochs=8000, learning_rate=0.005, batch_size=32, early_stop=False),
    #     DenseLayer(n_inputs=X.shape[1], n_neurons=20,  activation="sigmoid"),
    #     DenseLayer(n_inputs=20, n_neurons=20, activation="sigmoid"),
    #     DenseLayer(n_inputs=20, n_neurons=2, activation="softmax"),
    # ])
    model.summary()

    print(warning("\nTraining neural network model..."))
    model.train()

    print(warning("\nSaving neural network model in trained.json..."))
    model.save_model_as_json('trained.json')

    print(("Successfully trained model ✨"))
