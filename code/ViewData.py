import pickle

# Load the saved weights and biases
with open('model_weights_and_biases.pkl', 'rb') as f:
    weights_biases = pickle.load(f)

print(weights_biases)
