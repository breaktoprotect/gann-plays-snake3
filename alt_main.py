import numpy as np 
import time
from gann_agent_alternative import *

import matplotlib.pyplot as plt
import uuid
import os

def main():
    #* Instantiate Snake Agent
    initial_population_size = 10000
    population_size = 500
    crossover_rate = 0.8
    mutation_rate = 0.1
    elements_mutation_rate = 0.01
    num_of_processes = 6 # simultaneous evaluation processes
    height = 12
    width = 12
    gann_player = GANNAgent(initial_population_size=initial_population_size,population_size=population_size, crossover_rate=crossover_rate, mutation_rate=mutation_rate, elements_mutation_rate=elements_mutation_rate, 
    nn_shape=(33,20,12,4), num_of_processes = num_of_processes, env_height=height, env_width=width)
    #nn_shape=(33,40,24,4), num_of_processes = num_of_processes, env_height=height, env_width=width) #!experiment with hidden layers (double)

    #? Optional: Watch Saved Snake
    #watch_saved_snake('bab6b2bb98fd443dbff3dfc91b1bd1f6/gen340_best_snake.npy', gann_player, num_of_times=5, frequency=50)
    #return

    #* For graph visualization
    generations_list = []
    best_fitness_list = []
    average_fitness_list = []
    best_game_score_list = []
    plt.title("Fitness over Generations\nEnvironment: {HEIGHT} x {WIDTH} with Population: {POP_SIZE}\nCrossover Rate: {CROSSOVER_RATE} / Mutation Rate: {MUTATION_RATE} / Elements Mutation Chance:{ELM_RATE}".format(POP_SIZE=population_size, HEIGHT=height, WIDTH=width,CROSSOVER_RATE=crossover_rate,MUTATION_RATE=mutation_rate, ELM_RATE=elements_mutation_rate))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    #plt.ylim(0, len(target_phrase))

    # Pre-plot
    plt.plot(generations_list, best_fitness_list, 'ro', label="Best Fit Snake (Log Base 2)", color="red")
    plt.plot(generations_list, average_fitness_list, label='Average Population Fitness (Log Base 2)', color="orange")
    plt.plot(generations_list, best_game_score_list, marker='o', label='Best Game Score', color="blue")
    plt.legend(loc="upper left")

    #* Create a folder to store best snakes in each gen and list of average_fitness and best_fitness
    state_uuid = uuid.uuid4().hex
    os.mkdir(state_uuid)

    #* Actual Evolution - Generation starts from 0 
    # 0 - randomized
    # 1 - evolutioned
    for i in range(0,2000): # Make 2000 is the max generation
        current_best_snake, best_score, average_score, average_game_score = gann_player.evolve_population()

        #? Plotting the generation/fitness graph
        print("[*] Plotting results...", end="", flush=True)
        generations_list.append(i)
        best_fitness_list.append(np.log2(best_score)) # To make graph looks nicer
        average_fitness_list.append(np.log2(average_score))
        best_game_score_list.append(average_game_score)
        plt.plot(generations_list, best_fitness_list, 'ro', label="Best Fit Snake (Log Base 2)", color="red")
        plt.plot(generations_list, average_fitness_list, label='Average Population Fitness (Log Base 2)', color="orange")
        plt.plot(generations_list, best_game_score_list, marker='o', label='Best Game Score', color="blue")
        
        #? Update plotting
        if i % 1 == 0:
            plt.pause(0.001)
        print("OK")
        print("")

        #* Saving state
        print("[*] Saving state...", end="", flush=True)
        np.savetxt(state_uuid + '/best_fitness_list.txt', best_fitness_list)
        np.savetxt(state_uuid + '/average_fitness_list.txt', average_fitness_list)
        np.savetxt(state_uuid + '/best_game_score_list.txt', best_game_score_list)
        gann_player.save_snake(current_best_snake, state_uuid + "/gen{GEN}_best_snake".format(GEN=i))
        plt.savefig(state_uuid + '/a-fitness-over-generations-graph.png')
        print("OK")

#* Watch Saved snake
def watch_saved_snake(filename, gann_player, num_of_times=5, frequency=50):
    saved_snake = gann_player.load_snake(filename)
    for games in range(0, num_of_times):
        gann_player.evaluate_snake_model(saved_snake, render=True,frequency=frequency)
        time.sleep(1)
    return


        

        
   
if __name__ == "__main__":
    main()