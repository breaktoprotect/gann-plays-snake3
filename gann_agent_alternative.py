# Author        : JS @breaktoprotect
# Date started  : 12 August 2020
# Description:
# Alternate approach Instead of randomizing and then fit, generate random models then retain fit ones

import gym
import numpy as np 
import gym_snake3 
#import gym_snake
import time
import random
import os
import statistics
import math
import keyboard

# Standard Deep Neural Network 
#import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected 
#from tflearn.layers.estimator import regression

# For graph
#import tensorflow as tf

# Simple Self-implemented Neural Networ (No biases, and only Feed-forward)
import simple_neural_network as snn

# For human training (validation of model)
import keyboard

class GANNAgent:
    def __init__(self, initial_population_size=2000 ,population_size=2000, crossover_rate=0.5, mutation_rate=0.01, nn_shape=(32, 20, 8, 4), env_width=20, env_height=20):
        # Game environment
        self.env = gym.make('snake3-v0', render=True, segment_width=25, width=env_width, height=env_height) 
        #self.env = gym.make('snake-v0', render=True)
        self.env.reset()

        # Genetic Algorithm
        self.initial_population_size = initial_population_size
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation = 0 # starts with 0, only turns 1 after initial randomized generation
        self.prev_snakes_scores_list = None
        self.current_best_fit_snake = None
        self.current_best_scoring_snake = None

        # Neural network
        self.nn_shape = nn_shape # 33 inputs, 20 neurons hidden layer 1, 8 neurons hidden layer 2, 4 outputs

    #* Create the a snake's brain initialized with random weights and biases of 0
    def _create_nn_model(self, weights_list=None):
        model = snn.NeuralNet(self.nn_shape[0], self.nn_shape[1], self.nn_shape[2], self.nn_shape[3])

        # Set the defined weights list
        if weights_list != None:
            model.set_weights(weights_list)
        else:
            # If no existing weights_list, create a new one
            weights_list = model.get_weights()

        return model, weights_list[0], weights_list[1], weights_list[2]

    def save_snake(self, snake, filename):
        weights_list = snake[0].get_weights()
        np.save(filename, weights_list)

        return

    def load_snake(self, filename):
        weights_list = np.load(filename, allow_pickle=True)

        return self._create_nn_model(weights_list=weights_list)

    def evolve_population(self):
        if self.generation == 0:
            print("[*] Gen 0: New life! Generating initial random population of {INIT_POP}...".format(INIT_POP=self.initial_population_size))
            snakes_list = self.generate_random_population(self.initial_population_size)

            # Evaluation current population of snakes
            print("[*] Gen 0: Evaluating the fitness of current population of snakes...")
            snakes_scores_list = self.evaluate_population_fitness(snakes_list)

            # Summary of population fitness
            best_fitness_score, average_score, best_game_score = self.display_summary_of_fitness(snakes_scores_list)

            # Record the advance of a generation
            self.prev_snakes_scores_list = snakes_scores_list
            self.generation += 1

            return self.current_best_fit_snake, best_fitness_score, average_score, best_game_score
        else:  
            # Use the latest snakes_scores_list
            snakes_scores_list = self.prev_snakes_scores_list

            #* Genetic Algorithm 
            # Selection
            print("[*] Gen {GEN}: Selecting parents...".format(GEN=self.generation))
            parents_pool = self.selection(snakes_scores_list)

            # Copy - Keep Strong Parents
            print("[*] Gen {GEN}: Copy parents to keep strong genes...".format(GEN=self.generation))
            copy_snakes_list = self.copy(parents_pool, self.population_size, self.crossover_rate)

            # Crossover with Mutation - Evolve Strong Parents
            print("[*] Gen {GEN}: Crossover parents with chance of mutation...".format(GEN=self.generation))
            crossover_snakes_list = self.crossover(parents_pool, self.population_size, self.crossover_rate)

            #? Optional variant (TODO) Mutation only without Crossover - Variant 2

            # Combine copied children and crossover children
            new_snakes_list = copy_snakes_list + crossover_snakes_list

            # Evaluation current population of snakes
            print("[*] Gen {GEN}: Evaluating the fitness of current population of snakes...".format(GEN=self.generation))
            snakes_scores_list = self.evaluate_population_fitness(new_snakes_list)

            # Summary of population fitness
            best_fitness_score, average_score, best_game_score = self.display_summary_of_fitness(snakes_scores_list)

            # Record the advance of a generation
            self.prev_snakes_scores_list = snakes_scores_list
            self.generation += 1

            return self.current_best_fit_snake, best_fitness_score, average_score, best_game_score

    def generate_random_population(self, pop_size):
        snakes_list = []

        for i in range(0, pop_size):
            snake_model, fc1, fc2, fc3 = self._create_nn_model()
            snakes_list.append([snake_model, fc1, fc2, fc3])

        return snakes_list

    # Get scores of all snakes in population
    def evaluate_population_fitness(self, snakes_list):
        snakes_scores_list = []

        for snake in snakes_list:
            fitness_score, game_score = self.evaluate_snake_model(snake) 
            snakes_scores_list.append([snake, fitness_score, game_score])

        return snakes_scores_list

    # Get score of 1 snake model and 1 game
    def evaluate_snake_model(self, snake, render=False, frequency=10):
        self.env.reset()
        game_score = 0
        prev_observation = []
        
        model = snake[0] # [snake_model, fc1, fc2, fc3]

        if render:
            self.env.init_window()

        for steps in range(0,9999999999):
            if render:
                self.env.render()
                time.sleep(1/frequency)

            # If space bar pressed, trigger render
            if keyboard.is_pressed('space'):
                self.env.init_window()
                render = True

            # If esc is pressed, trigger window close
            if keyboard.is_pressed('escape'):
                self.env.close_window()
                render = False

            # Action (Decision) making
            if len(prev_observation) > 0:
                #action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
                action = np.argmax(model.feed_forward(np.array(prev_observation)))
            else:
                action = self.env.action_space.sample()

            # Execute a step
            observation, reward, done, info = self.env.step(action)

            #debug
            #print("debug:: prev_observation:", prev_observation)

            #? (Not necessary) Record game memory
            if len(prev_observation) > 0:
                pass
                #cur_game_memory_list.append([prev_observation, action])
                #cur_choices_list.append(action)

            prev_observation = observation # normalized to a sequential 400 inputs (20 x 20)
            game_score += reward
            
            # Terminate when game has ended
            if done:
                break

        # Using a special fitness function obtained from https://chrispresso.coffee/2019/09/22/ai-learns-to-play-snake/
        fitness_score = self._calc_fitness(steps, game_score)

        #TODO: Alternative to playing only once, play 5 times and get the average score?

        return fitness_score, game_score
    
    # Fitness function taken from: https://chrispresso.coffee/2019/09/22/ai-learns-to-play-snake/
    #* Fitness function
    def _calc_fitness(self, steps, rewards):
        return steps + (2**rewards + 500*rewards**2.1)-(0.25*steps**1.3*rewards**1.2)

    #* Selection
    def selection(self, snakes_score_list):
        parents_pool = []

        #? Fitness score normalization
        # Fitness score of each snake can get very large (e.g 17179869184 at Gen300+)
        #1. Find the largest fitness score
        highest_fitness_score = 0
        for fitness in snakes_score_list:
            if fitness[1] > highest_fitness_score:
                highest_fitness_score = fitness[1]

        #debug
        print("Debug-> highest fitness score:", highest_fitness_score)
            
        #2. Determine the acceptable denominator
        divisor = 1
        while highest_fitness_score > 1000:
            highest_fitness_score = highest_fitness_score / 10
            divisor *= 10

        print("Debug-> Current divisor:", divisor)
        
        #! Experimental: Weed out of the lousy snakes
        #TODO

        #* Selection 
        # With 'divisor' in implementation, poor-performing snakes has a chance to be absolutely eliminated 
        for snake_score in snakes_score_list:
            for _ in range(0, round(snake_score[1]/divisor)): # Divisor to reduce the potentially large number
                parents_pool.append(snake_score[0]) # snake, which includes model, fc1 weights, fc2 weights, fc3 weights

        #debug
        print("parents_pool length:", len(parents_pool))

        return parents_pool

    #* Copy original parents (no crossover, no mutation)
    def copy(self, parents_pool, population_size, crossover_rate):
        new_snakes_list = []
        for i in range(0, round(population_size*(1-crossover_rate))):
            child_snake = parents_pool[random.randint(0, len(parents_pool)-1)].copy()
            new_snakes_list.append(child_snake)

        return new_snakes_list

    #* Crossover with Mutation
    def crossover(self, parents_pool, population_size, crossover_rate):
        new_snakes_list = []        

        # Calculate midpoints for individual fully-connected layers
        parent_1 = parents_pool[random.randint(0, len(parents_pool)-1)] # for reference, not to be used in crossover
        fc1_midpoint = math.ceil(len(parent_1[1])/2)
        fc2_midpoint = math.ceil(len(parent_1[2])/2)
        fc3_midpoint = math.ceil(len(parent_1[3])/2)
        
        #debug
        #print("fc1_midpoint:",fc1_midpoint)
        #print("fc2_midpoint:",fc2_midpoint)
        #print("fc3_midpoint:",fc3_midpoint)
        

        for i in range(0,round(population_size*crossover_rate)):
            # Randomly select parents from pool
            parent_1 = parents_pool[random.randint(0, len(parents_pool)-1)].copy()
            parent_2 = parents_pool[random.randint(0, len(parents_pool)-1)].copy()

            # New snake (child)
            child_fc1_w = []
            child_fc2_w = []
            child_fc3_w = []

            # Select half for each parents to combine
            # fc1 layer
            child_fc1_w = np.array(parent_1[1][:fc1_midpoint])
            child_fc1_w = np.vstack([child_fc1_w, parent_2[1][fc1_midpoint:]])

            # fc2 weights
            child_fc2_w = np.array(parent_1[2][:fc2_midpoint])
            child_fc2_w = np.vstack([child_fc2_w, parent_2[2][fc2_midpoint:]])

            # fc3 weights
            child_fc3_w = np.array(parent_1[3][:fc3_midpoint])
            child_fc3_w = np.vstack([child_fc3_w, parent_2[3][fc3_midpoint:]])

            #debug
            #print("child_fc1_w length:", len(child_fc1_w))
            #print("child_fc2_w length:", len(child_fc2_w))
            #print("child_fc3_w length:", len(child_fc3_w))
            #/debug

            #debug
            #print("child_fc1_w:", child_fc1_w)

            # Mutation
            for fc_weights in [child_fc1_w, child_fc2_w, child_fc3_w]:
                for x, w_l in enumerate(fc_weights):
                    for y, w_lol in enumerate(fc_weights[x]):
                        if random.random() < self.mutation_rate:
                            fc_weights[x][y] = random.uniform(-1,1) # Mutation by random number generated

            # Create child snake
            child_weights_list = [child_fc1_w, child_fc2_w, child_fc3_w]
            child_snake_model, fc1, fc2, fc3 = self._create_nn_model(weights_list=child_weights_list)

            new_snakes_list.append([child_snake_model, fc1, fc2, fc3])

        #debug
        #print("-> Finished crossover. Length of new_snakes_list: {LENGTH}".format(LENGTH=len(new_snakes_list)))
        #print("-> new_snakes_list[0]:", new_snakes_list[0])

        return new_snakes_list
    
    def display_summary_of_fitness(self, snakes_scores_list):
        # Header
        print("[*] ---------------------------------")

        # Generation info
        print("[*] Current generation: {GEN}".format(GEN=self.generation))

        #* Best fitness & Game score
        best_fitness_score = 0
        best_game_score = 0
        total_fitness_score = 0
        best_fit_snake_index = None
        best_scoring_snake_index = None
        for index, snake_score in enumerate(snakes_scores_list):
            total_fitness_score += snake_score[1]
            if snake_score[1] >= best_fitness_score:
                best_fitness_score = snake_score[1]
                best_fit_snake_index = index
            if snake_score[2] >= best_game_score:
                best_game_score = snake_score[2]
                best_scoring_snake_index = index

        #* Population Average Fitness
        average_score = total_fitness_score / len(snakes_scores_list)

        self.current_best_fit_snake = snakes_scores_list[best_fit_snake_index][0]
        self.current_best_scoring_snake = snakes_scores_list[best_scoring_snake_index][0]
        print("[+] Best fitness snake with rated: {FITNESS}".format(FITNESS=best_fitness_score))
        print("[+] Best scoring snake with score: {SCORE}".format(SCORE=best_game_score))
        print("[~] Average fitness of snake generation: {AVG}".format(AVG=average_score))
        print("") # new empty line

        '''
        # Population Average Game Score
        total_game_score = 0
        for snake_score in snakes_scores_list:
            total_game_score += snake_score[2]
        average_game_score = total_game_score / len(snakes_scores_list)
        '''

        return best_fitness_score, average_score, best_game_score 