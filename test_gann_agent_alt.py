from gann_agent_alternative import *
from simple_neural_network import NeuralNet

def test_main(gann_player, parent_1, parent_2):
    #test_spx_row(gann_player, parent_1, parent_2)

    #test_spx_col(gann_player, parent_1, parent_2)

    #test_clipped_snakes(gann_player, parent_1, parent_2)

    test_fitness(gann_player)

def test_fitness(gann_player):
    game_score_1 = 33
    game_steps = 2000 # Constant
    game_score_2 = round(33 * (1 - 0.05))

    print("Game score 1 fitness:",gann_player._calc_fitness(game_steps, game_score_1))
    print("Game score 2 fitness:", gann_player._calc_fitness(game_steps, game_score_2))

def test_spx_row(gann_player, parent_1, parent_2):
    parents_pool = [parent_1, parent_2]

    child_snakes = gann_player.singlepoint_crossover_row(parents_pool, 2)
    #print("child_snakes[0] weights\n:", child_snakes[0].get_weights())
    #print("child_snakes[1] weights\n:", child_snakes[1].get_weights())
    print("child_snakes[0] biases\n:", child_snakes[0].get_biases())
    print("child_snakes[1] biases\n:", child_snakes[1].get_biases())
    return

def test_spx_col(gann_player, parent_1, parent_2):
    parents_pool = [parent_1, parent_2]

    child_snakes = gann_player.singlepoint_crossover_col(parents_pool, 2)
    print("child_snakes[0] weights\n:", child_snakes[0].get_weights())
    print("child_snakes[1] weights\n:", child_snakes[1].get_weights())
    #print("child_snakes[0] biases\n:", child_snakes[0].get_biases())
    #print("child_snakes[1] biases\n:", child_snakes[1].get_biases())
    return

def test_clipped_snakes(gann_player, parent_1, parent_2):
    snake = gann_player.load_snake('elite_snakes/gen702_best_snake.npy')
    snake_list = [snake]

    print("old snake_list:\n", snake_list[0].get_weights())

    new_snake_list = gann_player._get_clipped_snakes(snake_list)

    print("clipped snake_list:\n", snake_list[0].get_weights())


if __name__ == "__main__":
    #* Instantiate Snake Agent
    initial_population_size = 1500
    population_size = 1500
    crossover_rate = 0.6667
    parental_genes_deviation_rate = 1.0 # Previously 1.0 #!don't matter now
    parental_genes_deviation_factor = 0.03 # previously 0.01, 0.05
    mutation_rate = 1 # Previously 0.9, 1.0
    gene_mutation_rate = 0.05 # Previously 0.01
    gaussian_mutation_scale = 0.2 # previously 0.1, 0.2
    num_of_processes = 6 # simultaneous evaluation processes
    height = 12
    width = 12
    gann_player = GANNAgent(initial_population_size=initial_population_size,population_size=population_size, crossover_rate=crossover_rate, mutation_rate=mutation_rate, gene_mutation_rate=gene_mutation_rate, 
    nn_shape=(32,20,12,4), num_of_processes = num_of_processes, env_height=height, env_width=width)

    #* Instantiate two parents and parents pool
    parent_1 = NeuralNet(32,20,12,4)
    parent_2 = NeuralNet(32,20,12,4)
    parent_1_w = parent_1.get_weights()
    parent_2_w = parent_2.get_weights()
    parent_1_b = parent_1.get_biases()
    parent_2_b = parent_2.get_biases()

    # Init distinct values for testing
    for l, _ in enumerate(parent_1_w):
        for i, x in enumerate(parent_1_w[l]):
            for j, y in enumerate(parent_1_w[l][i]):
                parent_1_w[l][i][j] = -2
                parent_2_w[l][i][j] = 2
    print("")

    for l, _ in enumerate(parent_1_b):
        for i, x in enumerate(parent_1_b[l]):
            parent_1_b[l][i] = -2
            parent_2_b[l][i] = 2

    parent_1.set_weights(parent_1_w)
    parent_1.set_biases(parent_1_b)
    parent_2.set_weights(parent_2_w)
    parent_2.set_biases(parent_2_b)



    # Enter main testing
    test_main(gann_player, parent_1, parent_2)