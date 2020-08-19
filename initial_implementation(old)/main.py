import numpy as np 
import time
from gann_agent import *

#!Experimental
import tensorflow as tf

#? Testing only debug
import gym
import gym_snake3

def main():
    #? Testing only debug
    '''
    env = gym.make('snake3-v0', render=True, segment_width=25, width=20, height=20)
    env.reset()
    print("normalized_distance:", env._normalized_distance_two_points((20,20), (19,20)))

    print("detect wall:", env._detect_object_type_at_a_point((0,1)))
    print("detect body:", env._detect_object_type_at_a_point(env.snake_segments[1]))
    print("detect apple:", env._detect_object_type_at_a_point(env.apple_pos))
    print("detect hopefully nothing:", env._detect_object_type_at_a_point((18,18))) # (19,19) is wall
    print("detect wall:", env._detect_object_type_at_a_point((1,1))) # Not wall, (0,0) is wall
    print("")
    

    env.apple_pos = (5,9)
    env.render()
    #print('sweep up:', env._sweep_direction_for_object(env.snake_segments[0], (0,-1)))
    #print('sweep top-right:', env._sweep_direction_for_object(env.snake_segments[0], (1,-1)))
    #print('sweep down:', env._sweep_direction_for_object(env.snake_segments[0], (0,1)))
    #print('sweep down', env._sweep_direction_for_object(env.snake_segments[0], (0,1)))
    print('_sense_all_directions():', env._sense_all_directions(env.snake_segments[0]))
    input()
    return
    '''

    #? Apple Spawn testing
    '''
    env = gym.make('snake3-v0', render=True, segment_width=25, width=20, height=20)
    env.reset()
    env.render()
    #env.snake_segments = [(5,5), (5,6), (5,7), (5,8), (5,9), (5,10), (5,11), (5,12), (6,12),(6,11),(7,11),(8,11), (9,11), (10,11)]
    #env._spawn_apple()
    
    print("env.snake_segments:",env.snake_segments)
    input()
    return
    '''

    #? Test tail direction
    '''
    env = gym.make('snake3-v0', render=True, segment_width=25, width=20, height=20)
    env.reset()
    env.snake_segments = [(5,5), (5,6), (5,7), (5,8), (5,9), (5,10), (5,11), (5,12), (6,12),(6,11),(7,11),(8,11), (9,11), (10,11), (10,10), (10,9)]
    env.render()
    print(env._sense_tail_direction())
    input()
    return
    '''

    #* Initialize GANN Agent
    gann_player = GANNAgent(fitness_top_percentile=0.0125, mutation_rate=0.05,learning_rate=1e-2) # 1e-3*5 0.005
    
    
    #* Human trained (visual testing of DNN model)
    #human_model = gann_player.human_training_data(num_of_epoch=30,verbose=True)
    '''
    # Load past cumulative list of training data from directory 
    human_model = gann_player.load_cumulative_training_data_and_model_fit('cumulative_training/', num_of_epoch=50)

    gann_player.play(num_of_games=10, frequency=200, model=human_model, random_game=False)
    
    return
    '''
    '''
    #* Genetic Algorithm Training!
    watch_every = 5

    for i in range(50):
        gann_player.generate_training_data(generation_population=2000, debug_render=False, debug_frequency=1,verbose=False, num_of_epoch=300)
        
        if ((i+1) % watch_every) == 0:
            gann_player.play(num_of_games=5)
    
    
    #* Manual Modeling of top candidates
    top_candidates = gann_player.load_cumulative_training_data_and_model_fit('top_candidates/', num_of_epoch=100)

    gann_player.play(num_of_games=10, frequency=200, model=top_candidates, random_game=False)
    '''
    #* Random game
    #gann_player.play(num_of_games=10, frequency=200, model=None, random_game=True)
    for _ in range(100):
        new_graph = tf.Graph()
        with new_graph.as_default():
            
            empty_model = gann_player._create_neural_network_model(33, 4)
            print(empty_model.get_weights(network.W))
            gann_player.play(num_of_games=1, model=empty_model,frequency=200, no_display=False)



if __name__ == "__main__":
    main()