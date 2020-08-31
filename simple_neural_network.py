# Author: JS @breaktoprotect
# Desc  : A simple feed-forward hardcoded two-hidden-layer neural network - only feed-forward, no backpropagation

import math
import random
import numpy as np
import copy

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
        self.input_h1_weights = copy.deepcopy(weights_list[0])
        self.h1_h2_weights = copy.deepcopy(weights_list[1])
        self.h2_output_weights = copy.deepcopy(weights_list[2])

        return

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
        self.hl1_biases = copy.deepcopy(biases_list[0])
        self.hl2_biases = copy.deepcopy(biases_list[1])
        self.output_biases = copy.deepcopy(biases_list[2])

        return

    # Make a copy
    def copy(self):
        new_nn = NeuralNet(self.input_size, self.h1_size, self.h2_size, self.output_size)

        new_nn.set_weights(self.get_weights())
        new_nn.set_biases(self.get_biases())

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
    np.set_printoptions(precision=5)
    nn = NeuralNet(2,8,6,4)
    inputs = [1,2]

    # Test
    '''
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
    '''
    # Parents
    parent_1 = NeuralNet(2,8,6,4)
    parent_2 = NeuralNet(2,8,6,4)
    parent_1_w = parent_1.get_weights()
    parent_2_w = parent_2.get_weights()
    parent_1_b = parent_1.get_biases()
    parent_2_b = parent_2.get_biases()

    # Init distinct values for testing
    for l, _ in enumerate(parent_1_w):
        for i, x in enumerate(parent_1_w[l]):
            for j, y in enumerate(parent_1_w[l][i]):
                parent_1_w[l][i][j] = 1
                parent_2_w[l][i][j] = 2

    for l, _ in enumerate(parent_1_b):
        for i, x in enumerate(parent_1_b[l]):
            parent_1_b[l][i] = 3
            parent_2_b[l][i] = 4

    # Test copy
    child = parent_1.copy()
    print(id(parent_1))
    print(parent_1.get_biases())

    new_biases = child.get_biases()
    new_biases[0] = 1
    child.set_biases(new_biases)
    print(id(child))
    print(child.get_biases())

    # Test multi point crossover
    '''
    child_snake = NeuralNet(2,8,6,4)
    child_snake_weights = child_snake.get_weights()
    for l, _ in enumerate(child_snake_weights):
        for i, x in enumerate(child_snake_weights[l]):
            if i == 0:
                # Select random midpoint
                split_point = random.randint(0, len(child_snake_weights[l][i])-1)

            for j, y in enumerate(child_snake_weights[l][i]):
                if j < split_point:
                    child_snake_weights[l][i][j] = parent_1_w[l][i][j]
                else:
                    child_snake_weights[l][i][j] = parent_2_w[l][i][j]

                child_snake_biases = child_snake.get_biases()
    child_snake_biases = child_snake.get_biases()
    for l, _ in enumerate(child_snake_biases):
        for i, x in enumerate(child_snake_biases[l]):
            if l == i:
                # Select random midpoint
                split_point = random.randint(0, len(child_snake_biases[l])-1)
            if i < split_point:
                child_snake_biases[l][i] = parent_1_b[l][i]
            else:
                child_snake_biases[l][i] = parent_2_b[l][i]
                
    print("child_snake_weights", child_snake_weights)
    print("child_snake_biases", child_snake_biases)
    '''

if __name__ == "__main__":
    main()
