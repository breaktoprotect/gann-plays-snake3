# Genetic Algorithm with Neural Network Snake Game Agent
## A completed game demo
![Agent Completion](https://github.com/breaktoprotect/gann-plays-snakev3/blob/master/demo/completed_GANN_agent.gif)

## Fitness/Score over Generations
![Score over generations](https://github.com/breaktoprotect/gann-plays-snakev3/blob/master/demo/a-fitness-over-generations-graph.png)

Note: It's not the complete one. The environment went into a loop as I earlier I did not taken into account game completion. The Snakes were retrained for another 100+ generations and many started clearing the game.

## Description
As part of learning the art of 'data science/machine learning/artificial intelligence', I embarked the journey to create an AI agent to play the classic Snake game. Using genetic algorithm to encourage evolution that ultimately convergences to optimal game outcome (full points).

## Learning Journey
- Write Neural Network and Feed Forward (prediction engine) from down up
- Code genetic algorithm from scratch (and help from Chrispresso)
- Write the Snake environment/game itself 
- Write test programs to validate the snake environment for correctness
- Write test programs to validate the Neural Network implementation
- Optimize to make the Snake environment 'play' or 'evaluate' the agent much faster (8x faster than original), and partially with the use of Numba

## Credits
Concepts, neural network structure and fitness functions are based on Chrispresso's work 
https://chrispresso.io/AI_Learns_To_Play_Snake
