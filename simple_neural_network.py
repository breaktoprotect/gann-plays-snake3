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

        #! Temporary: Simple Names
        '''
        self.input_layer_w = self.input_h1_weights
        self.hidden_layer_one_w = self.h1_h2_weights
        self.hidden_layer_two_w = self.h2_output_weights
        '''

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

    # Make a copy
    def copy(self):
        new_nn = NeuralNet(self.input_size, self.h1_size, self.h2_size, self.output_size)

        new_nn.set_weights(self.get_weights())

        return new_nn

    # FeedForward / Predict
    def feed_forward(self, inputs):
        if len(inputs) != self.input_size:
            print("[!] Fatal error performing feed_forward(). Input size not correct!")
            return
        
        #* Input to Hidden Layer One
        input_hl1_sums = np.dot(inputs, self.input_h1_weights)
        input_hl1_act = [self.relu(i) for i in input_hl1_sums] # Activation function results

        #debug
        #print("input_hl1_sums:", input_hl1_sums)
        #print("input_hl1_act",input_hl1_act)
        
        
        #* Hidden Layer One to Hidden Layer Two
        hl1_hl2_sums = np.dot(input_hl1_act, self.h1_h2_weights)
        hl1_hl2_act = [self.relu(i) for i in hl1_hl2_sums]

        #debug
        #print("hl1_hl2_sums", hl1_hl2_sums)
        #print("hl1_hl2_act", hl1_hl2_act)

        #* Hidden Layer Two to Output
        hl2_output_sums = np.dot(hl1_hl2_act, self.h2_output_weights)
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
    weights_list = nn.get_weights()

    print(">>> weights_list shape:", np.array(weights_list).shape)
    print(">>> weights_list:", weights_list)
    print(">>> nn shape:", nn.shape)
    print("")
    
    # Traverse throughout and set weights
    for l, _ in enumerate(weights_list):
        for i, x in enumerate(weights_list[l]):
            for j, y in enumerate(weights_list[l][i]):
                if random.uniform(0,1) < 0.5:
                    weights_list[l][i][j] = 1
                else:
                    weights_list[l][i][j] = 0

    print("*** changed weights_list:", weights_list)
    


if __name__ == "__main__":
    main()
