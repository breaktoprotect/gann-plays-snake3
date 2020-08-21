import numpy as np 
import time
from gann_agent_alternative import *

#* Instantiate Snake Agent
population_size = 500
mutation_rate = 0.00
height = 15
width = 15
gann_player = GANNAgent(population_size=population_size, mutation_rate=mutation_rate, nn_shape=(33,20,12,4),env_height=height, env_width=width)

def test():
    #? Testing model
    snakes_list = gann_player.generate_random_population(10)

    for i in snakes_list[:2]:
        print(i)

    return


    #? Testing model only
    
    gann_player = GANNAgent()
    
    model, fc1_w, fc2_w, fc3_w = gann_player._create_nn_model(33, 4)
    
    #print(model.get_train_vars())
    #fc1_weights = model.get_weights(fc1.W)

    #fc1_weights[0][0] = 1.0
    #model.set_weights(fc1.W, fc1_weights)

    midpoint = round(len(fc1_w)/2)
    print("half of fc1 weights:", fc1_w[:midpoint])
    print("")
    print("next half of fc1 weights:", fc1_w[midpoint:])
    
    for x, w_l in enumerate(fc1_w):
        for y, w_lol  in enumerate(fc1_w[x]):
            if random.random() < 0.01:
                    fc1_w[x][y] = random.uniform(-1,1)
    
    print("half of fc1 weights:", fc1_w[:midpoint])
    print("")
    print("next half of fc1 weights:", fc1_w[midpoint:])
    