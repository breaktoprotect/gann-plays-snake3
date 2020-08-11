# Author        : JS @breaktoprotect
# Date started  : 4 August 2020
# Description   : Attempting to implementic Genetic Algorithm with Neural Network to play Snake game (Gym Snake v3)
import gym
import numpy as np 
import gym_snake3 
#import gym_snake #!experimental
import time
import random
import tflearn
import os
import statistics

# Standard Deep Neural Network 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression

# For graph
import tensorflow as tf

# For human training (validation of model)
import keyboard

class GANNAgent:
    def __init__(self, fitness_top_percentile=0.05, segment_width=25, mutation_rate=0.02, learning_rate=1e-3, width=20, height=20):
        # Environment
        self.env = gym.make('snake3-v0', render=True, segment_width=25, width=width, height=height) 
        #self.env = gym.make('snake-v0', render=True) #!experimental
        self.env.reset()

        # Genetic Algorithm
        self.model_list = [None, None] # Model or the "brain" of the agent to make decisions; Only Two are loaded at the same time - Parents
        self.current_snake_model = None
        self.current_generation_index = 0 # Generation starts from 1. Only gets += after each generate_training_data()
        self.generation_fitness = [] # list of average fitness score per that generation
        self.generation_scores = [] # top 5 of sorted from highest to lowest snakes
        self.fitness_top_percentile = fitness_top_percentile
        self.learning_rate = learning_rate # or 0.01 (or 1e-3 0.001)
        self.mutation_rate = mutation_rate

    def generate_training_data(self, generation_population = 1000, acceptance_criteria = 0.05, verbose=False, debug_render=False, debug_frequency=100, num_of_epoch=5):
        training_data = []
        score_list = []
        accepted_score_list = []
        game_memory_lol = [] # game_memory list of lists

        if self.model_list[0] and self.model_list[0]:
            parent_model_1 = self.model_list[0] # Load the two parents model
            parent_model_2 = self.model_list[1]
            is_crossover = True
            print("[+] Models loaded with Two Parents from Generation: {GEN}".format(GEN=self.current_generation_index))
        else:
            #debug
            print("[*] No crossover. Randomizing inputs...")

            is_crossover = False

        print("[*] Generating training data...started.")

        #* Start of training each episode for defined number of population in the generation
        for episode in range(generation_population):
            #* Verbose
            if verbose and episode > 0:
                print("[*] Completion summary - {EPS} out of {TOTAL} - {COMPLETION}%".format(COMPLETION=episode/generation_population*100, EPS=episode, TOTAL=generation_population))

            #* Reset parameters for each episode
            self.env.reset()
            cur_game_memory_list = []
            cur_score = 0
            prev_observation = []
            cur_choices_list = []

            #* Decide if this snake agent is "mutated"
            if random.random() < self.mutation_rate:
                is_mutated = True
            else:
                is_mutated = False

            #* Run the game until it is done (using break)
            while True:
                if debug_render:
                    self.env.render()
                    time.sleep(1/debug_frequency)

                if is_crossover and len(prev_observation) > 0 and is_mutated == False:
                    #* Use current brain/model to predict action
                    self._get_parents_action(parent_model_1, parent_model_2, prev_observation)
                else:
                    # Random
                    action = self.env.action_space.sample()

                # Execute a step
                observation, reward, done, info = self.env.step(action)

                if verbose:
                    print(observation)
                    print("***action taken->", action)
                    print("")
                    
                
                #debug
                #print(observation)
                #print("")
                #time.sleep(1)
                #print("len(self._convert_board_to_inputs(observation)) -->", len(self._convert_board_to_inputs(observation)))
                #/debug

                # Record game memory
                if len(prev_observation) > 0:
                    cur_game_memory_list.append([prev_observation, action])
                    cur_choices_list.append(action)

                prev_observation = observation # normalized to a sequential 400 inputs (20 x 20)
                cur_score += reward

                # Terminate when game has ended
                if done:
                    break

            # Each end of game calculations
            score_list.append(cur_score)
            game_memory_lol.append(cur_game_memory_list)

            # For progression notification
            if (episode+1) % 500 == 0:
                print("[*]", episode+1, "has completed.")
        
        #? After each episode
        #* Pick top scorers and select for next generation
        print("[*] Calculating fitness of Generation {GENERATION_INDEX}...".format(GENERATION_INDEX=self.current_generation_index))
        minimum_score_requirement = self.fitness_min_score(score_list)

        #! Deprecated
        '''
        #debug
        print("debug: minimum_score_requirement ->",minimum_score_requirement)

        for index, score in enumerate(score_list):
            if score >= minimum_score_requirement:
                # Debug
                print("[+] Generation {GEN} Snake {INDEX} has been selected for score:{SCORE}.".format(GEN=self.current_generation_index,INDEX=index,SCORE=score))

                # Add to list of accepted scores
                accepted_score_list.append(score)

                # Add observations and output to training data
                for data in game_memory_lol[index]:
                    if data[1] == 0: # data[1] is action; where data[0] is prev_observation
                        output = [1, 0, 0, 0] # Up
                    elif data[1] == 1:
                        output = [0, 1, 0, 0] # Right
                    elif data[1] == 2:
                        output = [0, 0, 1, 0] # Down
                    elif data[1] == 3:
                        output = [0, 0, 0, 1] # Left

                    # Training data instance is 1) Previous observaiton; 2) Action carried out
                    training_data.append([data[0], output])
        '''
        # Take top 2 as parents
        tmp_score_list = score_list.copy()
        tmp_score_list.sort(reverse=True) # Highest to lowest
        index_of_score1 = score_list.index(tmp_score_list[0])
        index_of_score2 = score_list.index(tmp_score_list[1])
        indexes_of_score = [index_of_score1, index_of_score2]

        # Add to accepted_score_list for record
        accepted_score_list.append(tmp_score_list[0])
        accepted_score_list.append(tmp_score_list[1])

        #debug
        print("debug: index_of_score1:", index_of_score1)
        print("debug: index_of_score2:", index_of_score2)
        

        # Add observations and output to training data
        for i, list_index in enumerate(indexes_of_score):
            training_data = []
            for data in game_memory_lol[list_index]:
                if data[1] == 0: # data[1] is action; where data[0] is prev_observation
                    output = [1, 0, 0, 0] # Up
                elif data[1] == 1:
                    output = [0, 1, 0, 0] # Right
                elif data[1] == 2:
                    output = [0, 0, 1, 0] # Down
                elif data[1] == 3:
                    output = [0, 0, 0, 1] # Left

                # Training data instance is 1) Previous observaiton; 2) Action carried out
                training_data.append([data[0], output])

            # Perform neural network fit (training the agent's brain)
            self.model_list[i] = self._train_fitted_model(training_data, num_of_epoch=num_of_epoch)

        #debug
        print("debug** self.model_list length:", len(self.model_list))

        # Current Snake Model Takes after the Stronger Parent
        self.current_snake_model = self.model_list[0]

        # Perform neural network fitting (or training the agent's "brain")
        #self.model_list.append(self._train_fitted_model(training_data, num_of_epoch=num_of_epoch))

        # Update generation
        self._update_generation_information(accepted_score_list)

        # Select new parents to create future generation
        #self.parent_index_1, self.parent_index_2 = self._select_parents(minimum_generation=10)
        
        # Summary of each generation
        print("[*] Summary of Generation {GEN}".format(GEN=self.current_generation_index))
        accepted_score_list.sort(reverse=True)
        print("    Top scores(*):", accepted_score_list[:5])
        print("    Average scores(~):", self.generation_fitness)
        accepted_score_list.sort()
        print("    Minimum scores(-):", accepted_score_list[:5])

        return training_data

    # Update generation information
    def _update_generation_information(self, accepted_score_list):
        self.current_generation_index += 1
        self.generation_scores.append(accepted_score_list)
        self.generation_fitness.append(self.calculate_topscorers_fitness(accepted_score_list)) 
        return

    #? Human Training 
    def human_training_data(self, num_of_epoch=3,verbose=False):
        training_data = []
        self.env.reset()
        game_memory = []
        prev_observation = []
        score = 0

        # Start the game loop
        while True:
            action = -1 # Reset to invalid values first
            self.env.render()

            #* Enforce human player to play the game - no auto move
            while action < 0:
                time.sleep(0.1)

                if keyboard.is_pressed('up'):
                    action = 0
                elif keyboard.is_pressed('right'):
                    action = 1
                elif keyboard.is_pressed('down'):
                    action = 2
                elif keyboard.is_pressed('left'):
                    action = 3

            observation, reward, done, info = self.env.step(action)

            if verbose:
                print(observation)
                print("-> Action Taken: {ACTION}; Reward: {REWARD}; Done:{DONE}; Score: {SCORE}; Game Length: {LENGTH}".format(ACTION=action, REWARD=reward, DONE=done, SCORE=score, LENGTH=len(game_memory)))            

            # Log the game memory
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            #* End of training
            if done:
                print("[+] Human training completed.")
                break
            
        #* Prepare training data
        for data in game_memory:
            if data[1] == 0: # data[1] is action; where data[0] is prev_observation
                output = [1, 0, 0, 0] # Up
            elif data[1] == 1:
                output = [0, 1, 0, 0] # Right
            elif data[1] == 2:
                output = [0, 0, 1, 0] # Down
            elif data[1] == 3:
                output = [0, 0, 0, 1] # Left
            
            training_data.append([data[0], output])

        #debug
        print("debug training_data:", training_data[:5][0])

        #* Perform neural network fit (training the agent's brain)
        human_model = self._train_fitted_model(training_data, num_of_epoch=num_of_epoch)

        #* Save to file
        #training_data_save = np.array(training_data)
        np.save('human_training_data_on_{HEIGHT}_by_{WIDTH}_{RAND}'.format(RAND=str(random.randint(100000,999999)), WIDTH=self.env.ENV_WIDTH, HEIGHT=self.env.ENV_HEIGHT), training_data)
        
        return human_model

    # Load saved model
    def load_training_data_and_model_fit(self, filename, num_of_epoch=5):
        # Load training data
        training_data = np.load(filename, allow_pickle=True)

        # Fit the model
        model = self._train_fitted_model(training_data, num_of_epoch)       
        
        return model

    # Load a list of trainings and combine them to fit
    def load_cumulative_training_data_and_model_fit(self, directory_path, num_of_epoch=5):
        # Load list of training data files
        combined_training_data = []

        for filename in os.listdir(directory_path):
            if ".npy" in filename:
                # Combine them into a single training data set
                combined_training_data.extend(np.load(directory_path + filename, allow_pickle=True)[:-3]) #[:-1] to exclude "death data"

                #debug
                print("debug:: combined with:", directory_path, filename)
         
        # Fit the model 
        model = self._train_fitted_model(combined_training_data, num_of_epoch)  

        return model

    #* Train the model / brain
    def _train_fitted_model(self, training_data, num_of_epoch, model=False): 
        X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1) # Observations
        y = [i[1] for i in training_data]

        # Create a new graph / session
        #new_graph = tf.Graph()
        #with new_graph.as_default(): #? working :)
        if not model:
            model = self._create_neural_network_model(input_size = len(X[0]), output_size= len(y[0]))

        #early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.80) #! experimental
        model.fit({'input': X}, {'targets': y}, n_epoch=num_of_epoch,snapshot_step=1000, show_metric=True, run_id='gann_agent') # snapshot_epoch=False, callbacks=early_stopping_cb) #!experimental

        return model

    # Defining the neural network
    def _create_neural_network_model(self, input_size, output_size):
        network = input_data(shape=[None, input_size, 1], name='input')

        network = fully_connected(network, 512, activation="relu") # activation function = rectified linear
        network = dropout(network,0.8) # 80% retention

        network = fully_connected(network, 512, activation="relu") # activation function = rectified linear
        network = dropout(network,0.8) # 80% retention
    
        network = fully_connected(network, output_size, activation='softmax') # 4 outputs or actions of agent
        network = regression(network, optimizer='adam', batch_size=16, learning_rate=self.learning_rate, loss='mean_square', name='targets')

        #network = regression(network, batch_size=16, loss='mean_square', name='targets')

        model = tflearn.DNN(network, tensorboard_dir='log')

        return model

    def get_current_generation_index(self):
        return self.current_generation_index

    def set_current_generation_index(self, generation_number):
        #TODO truncate/reset old generation

        if generation_number >= self.current_generation_index:
            print("[-] Generation number must be smaller than current generation.")
            return False

        self.current_generation_index = generation_number
        return True
    
    # Average fitness of current selected generation 
    def calculate_topscorers_fitness(self, accepted_score_list):
        average_topscorers_fitness = sum(accepted_score_list) / len(accepted_score_list)
        return average_topscorers_fitness

    # Fitness criteria defined as top percentile
    def fitness_min_score(self, score_list):
        index_of_min_score = round(len(score_list)*self.fitness_top_percentile)

        # High score to lowest score, starting index 0 to last 
        score_list.sort(reverse=True)

        min_score = score_list[index_of_min_score]

        #debug
        #print("score_list:",score_list)
        #print("index_of_min_score:",index_of_min_score)
        #print("sorted score_list:",score_list)

        return min_score

    def most_fit_generation_index(self):
        if not len(self.generation_fitness) > 0:
            print("[!] Cannot find most fit generation. Currently generation is less than 1.")
            return

        gf_list = self.generation_fitness.copy()

        gf_list.sort(reverse=True)

        return self.generation_fitness.index(gf_list[0])


    def crossover(self):
        pass

    # Use a single model to predict the next move
    def _get_action(self, model, prev_observation):
        action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
        #action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, 20, 20, 1))[0]) #!experimental

        return action
    
    # Use current two parent models (brains) to predict the next move
    def _get_parents_action(self, parent_model_1, parent_model_2, prev_observation):
        # 50% chance of either parent 1 or parent 2
        if random.random() > 0.5: 
            action = np.argmax(parent_model_1.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
            #action = np.argmax(parent_model_1.predict(np.array(prev_observation).reshape(-1, 20, 20, 1))[0]) #!experimental
        else:
            action = np.argmax(parent_model_2.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
            #action = np.argmax(parent_model_2.predict(np.array(prev_observation).reshape(-1, 20, 20, 1))[0]) #!experimental
        return action
        
    #* Play the selected model based on generation index
    def play(self, model=None, num_of_games=5, frequency=50, random_game=False):
        # If model not defined, choose the best fitness generation
        if model:
            cur_model = model
        else:
            cur_model = self.current_snake_model

        # Init
        cur_score_list = []

        for episode in range(num_of_games):
            #* Reset parameters for each episode
            self.env.reset()
            #cur_game_memory_list = []
            cur_score = 0
            prev_observation = []
            #cur_choices_list = []

            #* In each game
            while True:
                time.sleep(1/frequency)
                self.env.render()

                if len(prev_observation) > 0 and not random_game:
                    #* Use current brain/model to predict action
                    action = self._get_action(cur_model, prev_observation)
                else:
                    # Random move if not yet have any observation
                    action = self.env.action_space.sample()

                # Execute a step
                observation, reward, done, info = self.env.step(action)

                prev_observation = observation 
                cur_score += reward

                if done:
                    time.sleep(1) # Pause for a second
                    break

            # End of current game
            cur_score_list.append(cur_score)
        
        # Display summary of results
        sorted_list = cur_score_list
        sorted_list.sort(reverse=True)
        average = statistics.mean(cur_score_list)
        print("[*] Summary of scores:",cur_score_list)
        print("    -> Highest score:", sorted_list[0])
        print("    -> Average score:", average)

        # Log model-played games
        try:
            np.loadtxt("model_played_records.csv")
            np.savetxt("model_played_records", cur_score_list)
        except:
            print("[-] No existing records as of yet.")
            np.savetxt("model_played_records.csv", cur_score_list)

# !experimental - doesn't work in this tflearn/tensorflow version
'''
class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh
    
    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration
        print("test, epoch ended!!!")
'''