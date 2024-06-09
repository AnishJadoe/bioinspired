import numpy as np


class NeuralNet():
    def __init__(self, n_inputs,n_outputs,n_hidden, chromosome):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_weights = self.n_inputs * self.n_hidden + self.n_outputs*self.n_hidden
        self.n_biases = self.n_hidden + self.n_outputs
        self.weights = self.get_weights(chromosome)
        self.biases = self.get_biases(chromosome)
        self.network = self.initialize_network()
        
    def get_weights(self,chromosome):
        weights = {}
        weights["input"] = np.array(chromosome[:self.n_inputs*self.n_hidden]).reshape(self.n_inputs,self.n_hidden)
        weights["output"] = np.array(chromosome[self.n_inputs*self.n_hidden:self.n_weights]).reshape(self.n_hidden,self.n_outputs)
        return weights

    def get_biases(self,chromosome):
        biases = {}
        biases["input"] = np.array(chromosome[self.n_weights:self.n_weights+self.n_hidden]).reshape(self.n_hidden,1)
        biases["output"] = np.array(chromosome[self.n_weights+self.n_hidden:]).reshape(self.n_outputs,1)
        return biases
    
    def initialize_network(self):
        network = dict()
        hidden_layer = {
            "weights": self.weights["input"],
            "biases": self.biases["input"],
        }
        network["hidden_layer"] = hidden_layer

        output_layer = {
            "weights": self.weights["output"],
            "biases": self.biases["output"],
        }
        network["output_layer"] = output_layer

        return network

    def forward_pass(self, input):
        """
        Starting from the input, this function does a forward pass to the output
        """

        v_j = self.calculate_v_j(input)
        y_jk = self.calculate_activation(v_j)
        
        output_weights = self.network["output_layer"]["weights"]  # w_jk
        output_bias = self.network["output_layer"]["biases"]  # b_k
        v_k = output_weights.T.dot(y_jk) + output_bias
        y_k = v_k

        return y_k
    
    def calculate_v_j(self, input):
        input_bias = self.network["hidden_layer"]["biases"]
        input_weights = self.network["hidden_layer"]["weights"]

        return input_weights.T.dot(input) + input_bias

    def calculate_activation(self, v_j):
        return (2 / (1 + np.exp(-2 * v_j))) - 1 # Tanh
    
    