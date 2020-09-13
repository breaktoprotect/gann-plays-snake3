import numpy as np 
import time
from gann_agent_alternative import *

import matplotlib.pyplot as plt
import uuid
import os

#debug
from simple_neural_network import NeuralNet

def main():
    #* Instantiate Snake Agent
    initial_population_size = 1500
    population_size = 1500 # Previously 1500
    crossover_rate = 0.75
    parental_genes_deviation_rate = 1.0 # Previously 1.0 
    parental_genes_deviation_factor = 0.05 # previously 0.01, 0.05, 0.03
    mutation_rate = 1 # Previously 0.9, 1.0
    gene_mutation_rate = 0.05 # Previously 0.01
    gaussian_mutation_scale = 0.2 # previously 0.1, 0.2
    num_of_processes = 6 # simultaneous evaluation processes
    height = 12
    width = 12
    gann_player = GANNAgent(initial_population_size=initial_population_size,population_size=population_size, crossover_rate=crossover_rate, mutation_rate=mutation_rate, gene_mutation_rate=gene_mutation_rate, 
    nn_shape=(32,20,12,4), num_of_processes = num_of_processes, env_height=height, env_width=width)
    #nn_shape=(33,40,24,4), num_of_processes = num_of_processes, env_height=height, env_width=width) #!experiment with hidden layers (double)
    
    #? Optional: Watch Saved Snake
    #watch_saved_snake('elite_snakes/pop750_score35_40_clipped.npy', gann_player, num_of_times=5, frequency=50)
    #return

    #* For graph visualization
    generations_list = []
    best_fitness_list = []
    average_fitness_list = []
    best_game_score_list = []
    average_game_score_list = []
    plt.style.use('dark_background') # Dark mode
    plt.title("Fitness over Generations\nEnvironment: {HEIGHT} x {WIDTH} with Population: {POP_SIZE}\nCrossover Rate: {CROSSOVER_RATE} / Mutation Rate: {MUTATION_RATE} / Elements Mutation Chance:{ELM_RATE}".format(POP_SIZE=population_size, HEIGHT=height, WIDTH=width,CROSSOVER_RATE=crossover_rate,MUTATION_RATE=mutation_rate, ELM_RATE=gene_mutation_rate))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    #plt.ylim(0, len(target_phrase))

    # Pre-plot
    plt.plot(generations_list, best_fitness_list, 'ro', label="Best Fit Snake (Log Base 2)", color="crimson")
    plt.plot(generations_list, best_game_score_list, marker='o', label='Best Game Score', color="dodgerblue")
    plt.plot(generations_list, average_fitness_list, label='Average Population Fitness (Log Base 2)', color="gold")
    plt.plot(generations_list, average_game_score_list, label="Average Population Score", color="lightsteelblue")
    
    plt.legend(loc="upper left")

    #* Create a folder to store best snakes in each gen and list of average_fitness and best_fitness
    state_uuid = "__" + uuid.uuid4().hex
    os.mkdir(state_uuid)

    #! Snakes Injection
    injected_snakes_path = ['elite_snakes/' + filename for filename in os.listdir('elite_snakes/')]
    gann_player.inject_snakes(injected_snakes_path) 

    #* Actual Evolution - Generation starts from 0 
    # 0 - randomized
    # 1 - evolutioned
    for i in range(0,2000): # Make 2000 is the max generation
        current_best_snake, best_fitness, average_fitness, best_game_score, average_score = gann_player.evolve_population() #return self.current_best_fit_snake, best_fitness_score, average_fitness, best_game_score, average_score

        #? Plotting the generation/fitness graph
        print("[*] Plotting results...", end="", flush=True)
        generations_list.append(i)
        best_fitness_list.append(np.log2(best_fitness)) # To make graph looks nicer
        average_fitness_list.append(np.log2(average_fitness))
        best_game_score_list.append(best_game_score)
        average_game_score_list.append(average_score)
        plt.plot(generations_list, best_fitness_list, 'ro', label="Best Fit Snake (Log Base 2)", color="crimson")
        plt.plot(generations_list, best_game_score_list, marker='o', label='Best Game Score', color="dodgerblue")
        plt.plot(generations_list, average_fitness_list, label='Average Population Fitness (Log Base 2)', color="gold")
        plt.plot(generations_list, average_game_score_list, label="Average Population Score", color="lightsteelblue")
        
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
        np.savetxt(state_uuid + '/average_game_score.txt', average_game_score_list)
        gann_player.save_snake(current_best_snake, state_uuid + "/gen{GEN}_best_snake".format(GEN=i))
        plt.savefig(state_uuid + '/a-fitness-over-generations-graph.png')
        print("OK")

#* Watch Saved snake
def watch_saved_snake(filename, gann_player, num_of_times=5, frequency=50):
    saved_snake = gann_player.load_snake(filename)
    snakes_score_list =[] # Required, but not used

    randomness_seed=random.randint(0,99999999)

    for games in range(0, num_of_times):
        gann_player.evaluate_snake_model(saved_snake, snakes_score_list, render=True,frequency=frequency, multiprocessing=False, randomness_seed=randomness_seed)
        time.sleep(1)
    return
   
if __name__ == "__main__":
    main()


'''
#! Debug testing for spx_r
    parent_1 = NeuralNet(32,20,12,4)
    parent_2 = NeuralNet(32,20,12,4)
    parent_1_w = parent_1.get_weights()
    parent_2_w = parent_2.get_weights()
    parent_1_b = parent_1.get_biases()
    parent_2_b = parent_2.get_biases()

    #? Init distinct values for testing
    for l, _ in enumerate(parent_1_w):
        for i, x in enumerate(parent_1_w[l]):
            for j, y in enumerate(parent_1_w[l][i]):
                parent_1_w[l][i][j] = 1
                parent_2_w[l][i][j] = 2
    print("")

    for l, _ in enumerate(parent_1_b):
        for i, x in enumerate(parent_1_b[l]):
            parent_1_b[l][i] = 3
            parent_2_b[l][i] = 4

    parent_1.set_weights(parent_1_w)
    parent_1.set_biases(parent_1_b)
    parent_2.set_weights(parent_2_w)
    parent_2.set_biases(parent_2_b)

    parents_pool = [parent_1, parent_2]

    child_snakes = gann_player.singlepoint_crossover_row(parents_pool, 2)
    #print("child_snakes[0] weights\n:", child_snakes[0].get_weights())
    #print("child_snakes[1] weights\n:", child_snakes[1].get_weights())
    print("child_snakes[0] biases\n:", child_snakes[0].get_biases())
    print("child_snakes[1] biases\n:", child_snakes[1].get_biases())
    return
    #!/debug
'''