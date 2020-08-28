# Author: JS @breaktoprotect
# Desc  : A simple feed-forward hardcoded two-hidden-layer neural network - only weights, no biases, no backpropagation

import math
import random
import numpy as np

class NeuralNet:
    def __init__(self, input_size, hidden_layer_one_size, hidden_layer_two_size, output_size):
        self.shape = (input_size, hidden_layer_one_size, hidden_layer_two_size, output_size)

        self.input_size = input_size
        self.h1_size = hidden_layer_one_size
        self.h2_size = hidden_layer_two_size
        self.output_size = output_size
        #self.shape = [input_size, hidden_layer_one_size, hidden_layer_two_size, output_size]

        #* Initialize all the layer-to-layer weights
        self.input_h1_weights = self._initialize_matrix(input_size, hidden_layer_one_size)
        self.h1_h2_weights = self._initialize_matrix(hidden_layer_one_size, hidden_layer_two_size)
        self.h2_output_weights = self._initialize_matrix(hidden_layer_two_size, output_size)

        #* Initialize all the various layers biases
        # self.input_biases = ... - no such a thing
        self.hl1_biases = self._initialize_bias(hidden_layer_one_size)
        self.hl2_biases = self._initialize_bias(hidden_layer_two_size)
        self.output_biases = self._initialize_bias(output_size)

    #* Weights
    # Initializes matrices of varied sizes
    def _initialize_matrix(self, rows, columns):
        matrix_list = []
        for r in range(0,rows):
            row = []
            for c in range(0,columns):
                row.append(random.uniform(-1,1))

            matrix_list.append(row)

        return np.array(matrix_list)

    # Get weights
    # [0] Input -> HL1, [1], HL1 -> HL2, [2], HL2 -> Output
    def get_weights(self): 
        weights_list = [self.input_h1_weights, self.h1_h2_weights, self.h2_output_weights]
    
        return weights_list

    # Set weights
    # [0] Input -> HL1, [1], HL1 -> HL2, [2], HL2 -> Output
    def set_weights(self, weights_list): 
        self.input_h1_weights = weights_list[0]
        self.h1_h2_weights = weights_list[1]
        self.h2_output_weights = weights_list[2]

        #! Just-in-case Test #debug remove later
        try:
            randomized_inputs = [random.uniform(-1,0) for i in range(self.input_size)]
            self.feed_forward(randomized_inputs)
        except:
            #debug
            print("Warning: Something went wrong in set_weights()!")
            return -1

        return 0

    #* Biases
    # Initialize biases of a layer
    def _initialize_bias(self, size):
        bias_list = []
        for _ in range(0, size):
            bias_list.append(random.uniform(-1,1))

        return np.array(bias_list)

    # Get biases
    def get_biases(self):
        biases_list = [self.hl1_biases, self.hl2_biases, self.output_biases]

        return biases_list

    # Set biases
    def set_biases(self, biases_list):
        self.hl1_biases = biases_list[0]
        self.hl2_biases = biases_list[1]
        self.output_biases = biases_list[2]

        return 0

    # Make a copy
    def copy(self):
        new_nn = NeuralNet(self.input_size, self.h1_size, self.h2_size, self.output_size)

        new_nn.set_weights(self.get_weights())

        return new_nn

    #* FeedForward / Predict
    def feed_forward(self, inputs):
        if len(inputs) != self.input_size:
            print("[!] Fatal error performing feed_forward(). Input size not correct!")
            return
        
        #* Input to Hidden Layer One
        input_hl1_sums = np.add(np.dot(inputs, self.input_h1_weights), self.hl1_biases) # Dot product + sum of biases
        input_hl1_act = [self.relu(i) for i in input_hl1_sums] # Activation function results
        
        #* Hidden Layer One to Hidden Layer Two
        hl1_hl2_sums = np.add(np.dot(input_hl1_act, self.h1_h2_weights), self.hl2_biases)
        hl1_hl2_act = [self.relu(i) for i in hl1_hl2_sums]

        #* Hidden Layer Two to Output
        hl2_output_sums = np.add(np.dot(hl1_hl2_act, self.h2_output_weights), self.output_biases)
        hl2_output_act = [self.sigmoid(i) for i in hl2_output_sums]
        output = hl2_output_act

        return output

    # Sigmoid Function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Rectified Linear Function
    def relu(self, x):
        return np.maximum(0,x)

#? Test bed only
def main():
    np.set_printoptions(precision=3)
    nn = NeuralNet(2,8,6,4)
    inputs = [1,2]

    # Test
    print("nn_feed_forward:", nn.feed_forward(inputs))
    print("")

    biases_list = nn.get_biases()
    print("nn.get_biases() BEFORE:",biases_list)
    print("")

    for i, layer in enumerate(biases_list):
        for j, bias in enumerate(biases_list[i]):
            biases_list[i][j] = 0.1
    nn.set_biases(biases_list)

    print("nn.get_biases() AFTER:",nn.get_biases())

if __name__ == "__main__":
    main()
