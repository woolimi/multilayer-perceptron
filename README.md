# Multi-layer Perceptron

Multi-layer perceptron is a ecole 42 project that introducing artificial neural networks.

The goal of this project is to implement a multilayer perceptron, in order to predict whether a cancer is malignant or benign on a dataset of breast cancer diagnosis in the Wisconsin.

## Dataset

It is a CSV file of 32 columns, the column diagnosis being the label you want to learn given all the other features of an example, it can be either the value `M` or `B` (for malignant or benign).

## Installation

```bash
pip install -r requirements.txt
# Data processing
python 01.prepare.py
# Train model
python 02.train.py
# Predict
python 03.predict.py
```

## Glossary of terms

### 1. Multilayer perceptron

A Multilayer Perceptron (MLP) is a type of artificial neural network (ANN) consisting of multiple layers of nodes (neurons), where each layer is fully connected to the next one. It is designed to approximate complex functions and solve problems that are not linearly separable.

#### Components:

- Input Layer: The layer that receives the input data.
- Hidden Layers: One or more intermediate layers where computations are performed. These layers allow the network to learn and model complex patterns in the data.
- Output Layer: The final layer that produces the output predictions.

#### Activation Functions:

Each neuron applies an activation function (e.g., ReLU, sigmoid, tanh) to its weighted input sum to introduce non-linearity into the model, enabling it to learn complex relationships.

### 2. Feedforward

Feedforward refers to the process where the input data passes through the neural network from the input layer to the output layer in a single direction.

- Input: The input data is fed into the input layer.
- Propagation: The data is propagated through each subsequent layer (hidden layers) by applying weights and activation functions.
- Output: The final layer produces the output prediction.

### 3. Backpropagation

Backpropagation (short for "backward propagation of errors") is a supervised learning algorithm used for training neural networks. It involves two main phases: forward pass and backward pass.

Process

1. Forward Pass: Calculate the output of the network by propagating the input through the layers.
2. Loss Calculation: Compute the loss (error) by comparing the predicted output with the actual target values using a loss function (e.g., mean squared error, cross-entropy).
3. Backward Pass: Propagate the error backward through the network to compute the gradient of the loss with respect to each weight by applying the chain rule of calculus.
4. Weight Update: Update the weights using the computed gradients to minimize the loss.

The goal of backpropagation is to optimize the weights of the neural network to reduce the error and improve the accuracy of predictions.

### 4. Gradient descent.

Gradient Descent is an optimization algorithm used to minimize the loss function in a neural network by iteratively adjusting the weights in the direction of the steepest descent of the loss function.

- Initialize Weights: Start with random initial weights.
- Compute Gradient: Calculate the gradient of the loss function with respect to the weights.
- Update Weights: Adjust the weights in the opposite direction of the gradient by a step size (**learning rate**).
- Repeat: Iterate the process until the loss function converges to a minimum value.

### 5. How to compute gradient descent in neural network ?

The backward pass involves computing the gradients of the loss function with respect to each weight in the network. This is done by applying **the chain rule of calculus** to propagate errors backward from the output layer to the input layer.

![notation](https://images.prismic.io/shortcut-french/Zpq6Mh5LeNNTxUL6_notation.png?auto=format,compress)

#### Steps of backpropagation

![backpropagation](https://images.prismic.io/shortcut-french/Zpq9OR5LeNNTxUMq_steps-backpropagation.png?auto=format,compress)

#### Output Layer (softmax) with cross-entropy loss

![softmax gradient descent with cross-entropy loss](https://images.prismic.io/shortcut-french/Zpq5ph5LeNNTxULp_softmax-gradient.png?auto=format,compress)

#### Hidden layer (Relu and sigmoid)

![Hidden layer gradient descent](https://images.prismic.io/shortcut-french/Zpq6xB5LeNNTxUMB_hidden-layer.png?auto=format,compress)

## Resources

- [Coursera machine learning - advanced learning algorithm](https://www.coursera.org/learn/advanced-learning-algorithms/home/week/1)
- [Neural Networks from Scratch in Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)

## Bonus

- Early stop

  ![Early stop image](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-28_at_12.59.56_PM_1D7lrVF.png)

- Evaluate the learning phase with multiple metrics: loss, accuracy, mse
