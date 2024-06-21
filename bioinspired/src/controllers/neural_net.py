import numpy as np

class NeuralNet:
    def __init__(self, n_inputs, n_outputs, n_hidden, chromosome):
        """
        Initializes the neural network.
        
        Parameters:
            n_inputs (int): Number of input neurons.
            n_outputs (int): Number of output neurons.
            n_hidden (int): Number of hidden neurons.
            chromosome (list): Genetic algorithm chromosome representing the weights and biases.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_weights = self.n_inputs * self.n_hidden + self.n_outputs * self.n_hidden
        self.n_biases = self.n_hidden + self.n_outputs
        
        self.weights, self.biases = self.decode_chromosome(chromosome)
        self.network = self.initialize_network()

    def decode_chromosome(self, chromosome):
        """
        Decodes the chromosome into weights and biases.
        
        Parameters:
            chromosome (list): Genetic algorithm chromosome.
            
        Returns:
            tuple: (weights, biases) dictionaries.
        """
        weights = {
            "input": np.array(chromosome[:self.n_inputs * self.n_hidden]).reshape(self.n_inputs, self.n_hidden),
            "output": np.array(chromosome[self.n_inputs * self.n_hidden:self.n_weights]).reshape(self.n_hidden, self.n_outputs)
        }
        biases = {
            "input": np.array(chromosome[self.n_weights:self.n_weights + self.n_hidden]).reshape(self.n_hidden, 1),
            "output": np.array(chromosome[self.n_weights + self.n_hidden:]).reshape(self.n_outputs, 1)
        }
        return weights, biases
    
    def initialize_network(self):
        """
        Initializes the neural network layers with weights and biases.
        
        Returns:
            dict: Network architecture with layers and corresponding weights and biases.
        """
        network = {
            "hidden_layer": {
                "weights": self.weights["input"],
                "biases": self.biases["input"],
            },
            "output_layer": {
                "weights": self.weights["output"],
                "biases": self.biases["output"],
            }
        }
        return network

    def forward_pass(self, input):
        """
        Performs a forward pass through the network.
        
        Parameters:
            input (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Output of the network.
        """
        v_j = self.calculate_v_j(input)
        y_jk = self.calculate_activation(v_j)
        
        output_weights = self.network["output_layer"]["weights"]
        output_bias = self.network["output_layer"]["biases"]
        v_k = output_weights.T.dot(y_jk) + output_bias
        y_k = v_k
        
        return y_k
    
    def calculate_v_j(self, input):
        """
        Calculates the input to the hidden layer neurons.
        
        Parameters:
            input (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Input to hidden layer neurons.
        """
        input_weights = self.network["hidden_layer"]["weights"]
        input_bias = self.network["hidden_layer"]["biases"]
        return input_weights.T.dot(input) + input_bias

    def calculate_activation(self, v_j):
        """
        Applies the activation function (tanh) to the input.
        
        Parameters:
            v_j (np.ndarray): Input to the hidden layer neurons.
            
        Returns:
            np.ndarray: Activated output.
        """
        return np.tanh(v_j)  # Using numpy's tanh function for clarity