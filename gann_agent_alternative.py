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
import multiprocessing as mp

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
    def __init__(self, initial_population_size=2000 ,population_size=2000, crossover_rate=0.5, mutation_rate=0.1, gene_mutation_rate=0.01, nn_shape=(32, 20, 8, 4), num_of_processes=4, env_width=20, env_height=20, apple_body_distance = False):
        # Game environment
        self.env = gym.make('snake3-v0', render=True, segment_width=25, width=env_width, height=env_height) 
        #self.env = gym.make('snake-v0', render=True)
        self.env.reset()

        # Game environment (multiprocessing)
        self.env_width = env_width
        self.env_height = env_height
        self.segment_width = 25
        self.apple_body_distance = apple_body_distance # Use boolean
        self.num_of_processes = num_of_processes

        # Genetic Algorithm
        self.initial_population_size = initial_population_size
        self.population_size = population_size

        self.parental_genes_deviation_rate = 0.9    # Deviation can happen during replication and crossover #TODO: un-hardcode it
        self.parental_genes_deviation_factor = 0.01 # +- value range to be randomized                       #TODO: un-hardcode it

        self.crossover_rate = crossover_rate
        self.uniform_crossover_rate = 0.5       #TODO: un-hardcode it
        self.singlepoint_crossover_rate = 0.5   #TODO: un-hardcode it

        self.mutation_rate = mutation_rate
        self.random_mutation_rate = 0.25 #TODO: un-hardcode it
        self.gene_mutation_rate = gene_mutation_rate 
        self.gaussian_mutation_rate = 0.75 #TODO: un-hardcode it
        self.gaussian_mutation_deviation = (1 - -1) * 0.05
        
        self.generation = 0 # starts with 0, only turns 1 after initial randomized generation
        self.prev_snakes_scores_list = None
        self.current_best_fit_snake = None
        self.current_best_scoring_snake = None

        # Neural network
        self.nn_shape = nn_shape # 32 inputs, 20 neurons hidden layer 1, 8 neurons hidden layer 2, 4 outputs

    #* Create the a snake's brain initialized with random weights and biases of 0
    def _create_nn_model(self, weights_list=None, biases_list=None):
        model = snn.NeuralNet(self.nn_shape[0], self.nn_shape[1], self.nn_shape[2], self.nn_shape[3])

        # Set the defined weights list
        if weights_list != None:
            model.set_weights(weights_list)

        # Set the defined biases list
        if biases_list != None:
            model.set_biases(biases_list)

        return model

    def save_snake(self, snake, filename):
        chromosome = []
        chromosome.append(snake.get_weights())
        chromosome.append(snake.get_biases())
        np.save(filename, chromosome)

        return

    def load_snake(self, filename):
        chromosome = np.load(filename, allow_pickle=True)
        weights_list = chromosome[0]
        biases_list = chromosome[1]

        return self._create_nn_model(weights_list=weights_list, biases_list=biases_list)

    def evolve_population(self):
        if self.generation == 0:
            print("[*] Gen 0: New life! Generating initial random population of {INIT_POP}...".format(INIT_POP=self.initial_population_size))
            snakes_list = self.generate_random_population(self.initial_population_size)

            # Evaluation current population of snakes
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
            parents_pool = self.selection(snakes_scores_list)

            # Replication - Keep Strong Parents
            print("[*] Gen {GEN}: Replicate parents to keep strong genes...".format(GEN=self.generation))
            replicated_snakes_list = self.replicate(parents_pool, self.population_size, self.crossover_rate)

            # Crossover with Mutation - Evolve Strong Parents
            print("[*] Gen {GEN}: Crossover parents with chance of mutation...".format(GEN=self.generation))
            crossover_snakes_list = self.get_crossovered_snakes(parents_pool, self.population_size, self.crossover_rate)

            #? Optional variant (TODO) Mutation only without Crossover - Variant 2

            # Combine copied children and crossover children
            new_snakes_list = replicated_snakes_list + crossover_snakes_list

            # Evaluation current population of snakes
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
            snake_model = self._create_nn_model()
            snakes_list.append(snake_model)

        return snakes_list

    #* Get scores of all snakes in population - multiprocessing implemented
    def evaluate_population_fitness(self, snakes_list):
        pool = mp.Pool(processes=self.num_of_processes)
        manager = mp.Manager()
        snakes_scores_list = manager.list()

        # Launch evaluations with multiprocessing
        for snake in snakes_list:
            pool.apply_async(self.evaluate_snake_model, args=(snake, snakes_scores_list))

        #debug
        total_pool_cache_len = len(pool._cache)
        while len(pool._cache) > 0:            
            print("[*] Gen {GEN}: Evaluating fitness of population...{CUR}/{TOTAL}\r".format(GEN=self.generation,CUR=total_pool_cache_len - len(pool._cache), TOTAL=total_pool_cache_len), end="")
            time.sleep(0.5)
        

        # Wait for all processes to complete
        pool.close()
        pool.join()

        return snakes_scores_list

    # Get score of 1 snake model and 1 game
    def evaluate_snake_model(self, snake, snakes_scores_list, multiprocessing=True, render=False, frequency=10):
        if multiprocessing:
            env = gym.make('snake3-v0', render=True, segment_width=25, width=self.env_width, height=self.env_height, apple_body_distance=self.apple_body_distance) 
        else:
            env = self.env
            
        env.reset()
        game_score = 0
        prev_observation = []
        
        model = snake 

        if render:
            env.init_window()

        for steps in range(0,9999999999):
            if render:
                env.render()
                time.sleep(1/frequency)

            # If space bar pressed, trigger render
            if keyboard.is_pressed('space'):
                env.init_window()
                render = True

            # If esc is pressed, trigger window close
            if keyboard.is_pressed('escape'):
                env.close_window()
                render = False

            # Action (Decision) making
            if len(prev_observation) > 0:
                #action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
                action = np.argmax(model.feed_forward(np.array(prev_observation)))
            else:
                action = env.action_space.sample()

            # Execute a step
            observation, reward, done, info = env.step(action)

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

        # Add to the snakes_scores_list
        snakes_scores_list.append([snake, fitness_score, game_score])

        return 
    
    # Fitness function taken from: https://chrispresso.coffee/2019/09/22/ai-learns-to-play-snake/
    #* Fitness function
    def _calc_fitness(self, steps, rewards):
        return steps + (2**rewards + 500*rewards**2.1)-(0.25*steps**1.3*rewards**1.2)

    #* Selection
    def selection(self, snakes_score_list):
        print("[*] Gen {GEN}: Selecting parents...".format(GEN=self.generation), end="", flush=True)
        parents_pool = []

        #? Fitness score normalization
        # Fitness score of each snake can get very large (e.g 17179869184 at Gen300+)
        #1. Find the largest fitness score
        highest_fitness_score = 0
        for fitness in snakes_score_list:
            if fitness[1] > highest_fitness_score:
                highest_fitness_score = fitness[1]

        #2. Determine the acceptable denominator
        divisor = 1
        while highest_fitness_score > 1000:
            highest_fitness_score = highest_fitness_score / 10
            divisor *= 10

        #* Selection 
        # With 'divisor' in implementation, poor-performing snakes has a chance to be absolutely eliminated 
        for snake_score in snakes_score_list:
            for _ in range(0, round(snake_score[1]/divisor)): # Divisor to reduce the potentially large number
                parents_pool.append(snake_score[0]) # snake, which includes model, fc1 weights, fc2 weights, fc3 weights

        #debug
        print("Total: {LENGTH}".format(GEN=self.generation,LENGTH=len(parents_pool)))

        return parents_pool

    #* Replicate original parents (no crossover, no mutation)
    def replicate(self, parents_pool, population_size, crossover_rate):
        new_snakes_list = []
        for i in range(0, round(population_size*(1-crossover_rate))):
            parent_snake = parents_pool[random.randint(0, len(parents_pool)-1)]
            child_snake = parent_snake.copy()

            # Chance to deviate
            if random.random() < self.parental_genes_deviation_rate:
                child_snake = self._deviate_genes(child_snake)

            new_snakes_list.append(child_snake)

            #debug - ensure unique objects are created
            #print("parent_snake id:", id(parent_snake))
            #print("child_snake id:", id(child_snake))

        return new_snakes_list

    # Genetic deviation - change the gene (w and b) of each child snake slightly (E.g. +- a small value) from parents
    def _deviate_genes(self, parent_snake):
        child_snake = parent_snake.copy()

        # Weights
        child_snake_weights = child_snake.get_weights()
        for l, _ in enumerate(child_snake_weights):
            for i, x in enumerate(child_snake_weights[l]):
                for j, y in enumerate(child_snake_weights[l][i]):
                    child_snake_weights[l][i][j] += random.uniform(-self.parental_genes_deviation_factor,self.parental_genes_deviation_factor) 

        # Biases
        child_snake_biases = child_snake.get_biases()
        for l, _ in enumerate(child_snake_biases):
            for i, x in enumerate(child_snake_biases[l]):
                child_snake_biases[l][i] += random.uniform(-self.parental_genes_deviation_factor,self.parental_genes_deviation_factor) 

        return child_snake

    #* Genetic Crossover of two parent snakes DNA
    def get_crossovered_snakes(self, parents_pool, population_size, crossover_rate):
        xo_snakes_list = []
        xo_pop_size = population_size * crossover_rate

        # Uniform Crossover
        xo_snakes_list += self.uniform_crossover(parents_pool, xo_pop_size * self.uniform_crossover_rate)

        # Singlepoint Crossover
        xo_snakes_list += self.singlepoint_crossover(parents_pool, xo_pop_size * self.singlepoint_crossover_rate)

        return xo_snakes_list

    '''
            self.uniform_crossover_rate = 0.5       #TODO: un-hardcode it
        self.singlepoint_crossover_rate = 0.5   #TODO: un-hardcode it

    def get_mutated_snake(self,snake):
        missed_chance = 0
        if random.random() < self.random_mutation_rate:
            return self.random_mutation(snake)
        else:
             missed_chance += self.random_mutation_rate

        if random.random() < (self.gaussian_mutation_rate + missed_chance):
            return self.gaussian_mutation(snake)

        #debug
        print("-------------------> get_mutated_snake missed_chance leak. Should have never reached here! <-----------------------")
        return
    '''

    # Get Two Parents' Weights and Biases
    def get_two_parents_specs(self, parents_pool):
        parent_1 = parents_pool[random.randint(0, len(parents_pool)-1)].copy()
        parent_2 = parents_pool[random.randint(0, len(parents_pool)-1)].copy()

        # Chance to deviate
        if random.random() < self.parental_genes_deviation_rate:
            parent_1 = self._deviate_genes(parent_1)
            parent_2 = self._deviate_genes(parent_2)

        parent_1_weights = parent_1.get_weights()
        parent_1_biases = parent_1.get_biases()

        parent_2_weights = parent_2.get_weights()
        parent_2_biases = parent_2.get_biases()

        return parent_1_weights, parent_1_biases, parent_2_weights, parent_2_biases

    # Uniform Crossover - Child obtains ~50% of DNA from two parents
    def uniform_crossover(self, parents_pool, crossover_size):
        new_snakes_list = []

        # 50% chance for each of two parents to assign weights to child
        for _ in range(0, round(crossover_size)):
            # Randomly select two parents from parents pool
            parent_1_weights, parent_1_biases, parent_2_weights, parent_2_biases = self.get_two_parents_specs(parents_pool)

            child_snake = snn.NeuralNet(self.nn_shape[0], self.nn_shape[1], self.nn_shape[2], self.nn_shape[3])

            # Weights
            child_snake_weights = child_snake.get_weights()

            for l, _ in enumerate(child_snake_weights):
                for i, x in enumerate(child_snake_weights[l]):
                    for j, y in enumerate(child_snake_weights[l][i]):
                        if random.uniform(0,1) < 0.5:
                            child_snake_weights[l][i][j] = parent_1_weights[l][i][j]
                        else:
                            child_snake_weights[l][i][j] = parent_2_weights[l][i][j]

            child_snake.set_weights(child_snake_weights)

            # Biases
            child_snake_biases = child_snake.get_biases()

            for l, _ in enumerate(child_snake_biases):
                for b, x in enumerate(child_snake_biases[l]):
                    if random.uniform(0,1) < 0.5:
                        child_snake_biases[l][b] = parent_1_biases[l][b]
                    else:
                        child_snake_biases[l][b] = parent_2_biases[l][b]

            child_snake.set_biases(child_snake_biases)

            #* Chance to have Mutations of different variations
            # Random, Gaussian
            if random.random() < self.mutation_rate:
                child_snake = self.get_mutated_snake(child_snake)
            
            # Add to the list
            new_snakes_list.append(child_snake)

        return new_snakes_list      

    # Single Point Crossover - Find a random midpoint and have two parents fill the gene before and after respectively
    def singlepoint_crossover(self, parents_pool, crossover_size):
        new_snakes_list = []

        # 50% chance for each of two parents to assign weights to child
        for _ in range(0, round(crossover_size)):
            # Randomly select two parents from parents pool
            parent_1_weights, parent_1_biases, parent_2_weights, parent_2_biases = self.get_two_parents_specs(parents_pool)

            child_snake = snn.NeuralNet(self.nn_shape[0], self.nn_shape[1], self.nn_shape[2], self.nn_shape[3])

            # Weights
            child_snake_weights = child_snake.get_weights()
            
            for l, _ in enumerate(child_snake_weights):
                for i, x in enumerate(child_snake_weights[l]):
                    if i == 0:
                        # Select random midpoint
                        split_point = random.randint(0, len(child_snake_weights[l][i])-1)

                    for j, y in enumerate(child_snake_weights[l][i]):
                        if j < split_point:
                            child_snake_weights[l][i][j] = parent_1_weights[l][i][j]
                        else:
                            child_snake_weights[l][i][j] = parent_2_weights[l][i][j]

            child_snake.set_weights(child_snake_weights)

            # Biases
            child_snake_biases = child_snake.get_biases()

            for l, _ in enumerate(child_snake_biases):
                for i, x in enumerate(child_snake_biases[l]):
                    if i == 0:
                        # Select random midpoint
                        split_point = random.randint(0, len(child_snake_biases[l])-1)

                    if i < split_point:
                        child_snake_biases[l][i] = parent_1_biases[l][i]
                    else:
                        child_snake_biases[l][i] = parent_2_biases[l][i]

            child_snake.set_biases(child_snake_biases)

            #* Chance to have Mutations of different variations
            # Random, Gaussian
            if random.random() < self.mutation_rate:
                child_snake = self.get_mutated_snake(child_snake)
            
            # Add to the list
            new_snakes_list.append(child_snake)

        return new_snakes_list     
    
    #* Genetic Mutation of the Snake's DNA
    def get_mutated_snake(self,snake):
        missed_chance = 0
        if random.random() < self.random_mutation_rate:
            return self.random_mutation(snake)
        else:
             missed_chance += self.random_mutation_rate

        if random.random() < (self.gaussian_mutation_rate + missed_chance):
            return self.gaussian_mutation(snake)

        #debug
        print("-------------------> get_mutated_snake missed_chance leak. Should have never reached here! <-----------------------")
        return

    #* Random Mutation
    def random_mutation(self, snake):
        new_snake_weights = snake.get_weights()
        new_snake_biases = snake.get_biases()

        # Weights
        for l, _ in enumerate(new_snake_weights):
            for i, x in enumerate(new_snake_weights[l]):
                for j, y in enumerate(new_snake_weights[l][i]):
                    if random.random() < self.gene_mutation_rate:
                        new_snake_weights[l][i][j] = random.uniform(-1,1) # Mutation by random number generated
        snake.set_weights(new_snake_weights)

        # Biases
        for l, _ in enumerate(new_snake_biases):
            for b, x in enumerate(new_snake_biases[l]):
                if random.random() < self.gene_mutation_rate:
                    new_snake_biases[l][b] = random.uniform(-1,1) # Mutation by random number generated
        snake.set_biases(new_snake_biases)

        return snake

    #* Gaussian Mutation
    # Standard deviation is by default 0.1 (between universal range of -1 to 1)
    def gaussian_mutation(self, snake):
        new_snake_weights = snake.get_weights()
        new_snake_biases = snake.get_biases()

        # Weights
        for l, _ in enumerate(new_snake_weights):
            for i, x in enumerate(new_snake_weights[l]):
                for j, y in enumerate(new_snake_weights[l][i]):
                    if random.random() < self.gene_mutation_rate:
                        new_snake_weights[l][i][j] = np.random.normal(loc=new_snake_weights[l][i][j], scale=self.gaussian_mutation_deviation) 
        snake.set_weights(new_snake_weights)

        # Biases
        for l, _ in enumerate(new_snake_biases):
            for b, x in enumerate(new_snake_biases[l]):
                if random.random() < self.gene_mutation_rate:
                    new_snake_biases[l][b] = np.random.normal(loc=new_snake_biases[l][b], scale=self.gaussian_mutation_deviation) 
        snake.set_biases(new_snake_biases)

        return snake
    
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
        print("    ---") 

        '''
        # Population Average Game Score
        total_game_score = 0
        for snake_score in snakes_scores_list:
            total_game_score += snake_score[2]
        average_game_score = total_game_score / len(snakes_scores_list)
        '''

        return best_fitness_score, average_score, best_game_score 