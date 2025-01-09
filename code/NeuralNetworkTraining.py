import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import pickle  # For loading model parameters

# Initialize nnfs for reproducibility
nnfs.init()

# Dense layer class: Handles forward and backward passes for fully connected layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation: Applies non-linearity, outputting max(0, input)
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation: Converts logits to probabilities
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Loss base class: Abstract class for loss functions
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

# Categorical Cross-Entropy: Calculates loss for classification problems
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true] if len(y_true.shape) == 1 else np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = self.dinputs / samples

# SGD Optimizer: Updates weights with Stochastic Gradient Descent
class Optimizer_SGD:
    def __init__(self, learning_rate=0.1, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        if self.momentum:
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

# Generate spiral dataset
X, y = spiral_data(samples=100, classes=3)

# Create layers and activations
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

# Load model parameters if available
try:
    with open('code/model_weights_and_biases.pkl', 'rb') as file:
        saved_model = pickle.load(file)
        dense1.weights = saved_model['dense1_weights']
        dense1.biases = saved_model['dense1_biases']
        dense2.weights = saved_model['dense2_weights']
        dense2.biases = saved_model['dense2_biases']
        print("Model parameters loaded successfully.")
except FileNotFoundError:
    print("No saved model found. Initializing a new model with random weights.")

loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.1, decay=1e-4, momentum=0.9)

# Training loop
epochs = 1500000
target_accuracy = 0.98
for epoch in range(epochs):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss and accuracy
    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update parameters
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"epoch: {epoch}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")

    # Check if target accuracy is reached
    if accuracy >= target_accuracy:
        print(f"Early stopping at epoch {epoch} with accuracy: {accuracy:.3f} and loss: {loss:.3f} ")
        with open('code/model_weights_and_biases.pkl', 'wb') as file:
            pickle.dump({
                'dense1_weights': dense1.weights,
                'dense1_biases': dense1.biases,
                'dense2_weights': dense2.weights,
                'dense2_biases': dense2.biases
            }, file)
        print("Model parameters saved.")
        break


'''
After training we have a reported accuracy of 0.98 and a loss of 0.083
'''