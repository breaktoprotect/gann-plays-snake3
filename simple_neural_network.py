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

        #* Test 
        try:
            randomized_inputs = [random.uniform(-1,0) for i in range(self.input_size)]
            self.feed_forward(randomized_inputs)
        except:
            return -1

        return 0


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
        hl1_hl2_act = [self.sigmoid(i) for i in hl1_hl2_sums]

        #debug
        #print("hl1_hl2_sums", hl1_hl2_sums)
        #print("hl1_hl2_act", hl1_hl2_act)

        #* Hidden Layer Two to Output
        output = np.dot(hl1_hl2_act, self.h2_output_weights)
        
        #debug
        #print("output:", output)

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
    
    print("Inputs:\n", inputs)    
    print("Input to H1 weights:\n",nn.input_h1_weights)
    print("")
    print("H1 to H2 weights:\n", nn.h1_h2_weights)
    print("")
    print("H2 to output weights\n", nn.h2_output_weights)
    print("")


    '''
    print("nn.feed_forward():", nn.feed_forward(inputs))
    print("np.argmax the results:", np.argmax(nn.feed_forward(inputs)))
    '''

    # Test set weights
    weights_list = nn.get_weights()

    print("weights_list[0]:", weights_list[0])
    print("")

    #weights_list[0][0] = 123
    #nn.set_weights(weights_list)

    #print("nn new weights_list[0]:", nn.get_weights())

    shape = (2,3,3,2)
    fc1_midpoint = math.ceil(len(nn.get_weights()[0])/2)
    fc2_midpoint = math.ceil(len(nn.get_weights()[1])/2)
    fc3_midpoint = math.ceil(len(nn.get_weights()[2])/2)

    print("weights [0] [0:fc1_midpoint]:", weights_list[0][:fc1_midpoint])
    print("weights [0] [fc1_midpoint:]:",weights_list[0][fc1_midpoint:])
    print("")
    print("re-combined weights [0]:", np.vstack([weights_list[0][:fc1_midpoint], weights_list[0][fc1_midpoint:]]))

    if np.all(weights_list[0] == np.vstack([weights_list[0][:fc1_midpoint], weights_list[0][fc1_midpoint:]])):
        print("[+] YEah, they're the same!")
    #new_nn = NeuralNet(shape[0], shape[1], shape[2], shape[3])


if __name__ == "__main__":
    main()
