import numpy as np
import matplotlib.pyplot as plt
import pickle
from nnfs.datasets import spiral_data
import nnfs

# Load the saved model weights and biases
def load_model():
    try:
        with open('code/model_weights_and_biases.pkl', 'rb') as file:
            saved_model = pickle.load(file)
            print("Model parameters loaded successfully.")
            return saved_model
    except FileNotFoundError:
        print("No saved model found.")
        return None

# Neural network layers and activations
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Load the dataset
X, y = spiral_data(samples=300, classes=3)

# Initialize nnfs for reproducibility
nnfs.init()

# Load saved model parameters
saved_model = load_model()

if saved_model:
    # Create the layers and activations
    dense1 = Layer_Dense(2, 64)
    dense2 = Layer_Dense(64, 3)
    
    # Assign the saved weights and biases
    dense1.weights = saved_model['dense1_weights']
    dense1.biases = saved_model['dense1_biases']
    dense2.weights = saved_model['dense2_weights']
    dense2.biases = saved_model['dense2_biases']
    
    activation1 = Activation_ReLU()
    activation2 = Activation_Softmax()

    # Perform a forward pass to get predictions
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    predictions = np.argmax(activation2.output, axis=1)

    # Plot the original spiral data
    plt.figure(figsize=(12, 6))

    # Plot the original spiral data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30)
    plt.title("Original Spiral Data")

    # Plot the neural network's categorized data
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis', s=30)
    plt.title("Neural Network Categorized Spiral Data")

    plt.show()

else:
    print("Unable to load the model. Please ensure the saved weights file exists.")
