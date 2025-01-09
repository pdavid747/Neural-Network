import numpy as np
import pickle
from nnfs.datasets import spiral_data
import nnfs

# Initialize nnfs for reproducibility
nnfs.init()

# Neural network components
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Load the saved model parameters
def load_model():
    try:
        with open('code/model_weights_and_biases.pkl', 'rb') as file:
            saved_model = pickle.load(file)
            print("Model parameters loaded successfully.")
            return saved_model
    except FileNotFoundError:
        print("No saved model found.")
        return None

# Testing individual components
def test_layer_dense():
    print("Testing Layer_Dense...")
    layer = Layer_Dense(2, 3)
    inputs = np.array([[1.0, 2.0]])
    layer.forward(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {layer.output}")
    assert layer.output.shape == (1, 3), "Output shape is incorrect for Layer_Dense."

def test_activation_relu():
    print("Testing Activation_ReLU...")
    activation = Activation_ReLU()
    inputs = np.array([[-1.0, 0.0, 1.0]])
    activation.forward(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {activation.output}")
    assert np.array_equal(activation.output, [[0.0, 0.0, 1.0]]), "ReLU activation failed."

def test_activation_softmax():
    print("Testing Activation_Softmax...")
    activation = Activation_Softmax()
    inputs = np.array([[1.0, 2.0, 3.0]])
    activation.forward(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {activation.output}")
    assert np.isclose(np.sum(activation.output), 1.0), "Softmax probabilities do not sum to 1."

# End-to-end test: Perform inference and validate predictions
def test_end_to_end():
    print("Testing end-to-end inference...")
    X, _ = spiral_data(samples=5, classes=3)  # Ensure that we use 15 samples initially
    
    # Slice X to ensure we have only the first 5 samples
    X = X[:5]
    
    print(f"Shape of X after slicing: {X.shape}")  # Debugging: print the shape of X
    
    # Check if the shape of X is correct (5, 2)
    assert X.shape == (5, 2), f"Expected input shape (5, 2), but got {X.shape}"
    
    saved_model = load_model()
    if not saved_model:
        print("Skipping end-to-end test: Model not found.")
        return

    # Recreate the network with saved weights and biases
    dense1 = Layer_Dense(2, 64)
    dense2 = Layer_Dense(64, 3)
    dense1.weights = saved_model['dense1_weights']
    dense1.biases = saved_model['dense1_biases']
    dense2.weights = saved_model['dense2_weights']
    dense2.biases = saved_model['dense2_biases']

    activation1 = Activation_ReLU()
    activation2 = Activation_Softmax()

    # Perform a forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Debugging: Check the shapes
    print(f"activation2.output.shape: {activation2.output.shape}")
    
    predictions = np.argmax(activation2.output, axis=1)
    print(f"Predictions: {predictions}")
    print(f"Predictions shape: {predictions.shape}")

    # Ensure that predictions have the correct shape (5,)
    assert predictions.shape == (5,), "Incorrect prediction shape in end-to-end test."

# Run tests
if __name__ == "__main__":
    test_layer_dense()
    test_activation_relu()
    test_activation_softmax()
    test_end_to_end()
    print("All tests completed.")
