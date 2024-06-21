"""
By running this file, the whole algorithm is run and plots
are made of the given paremeters
"""
# Loading in functions
import numpy as np
from src.robots.robots import TankNeuroRobot
from src.runners.genetic_algorithm import GeneticAlgorithmRunner
from src.utility.constants import *
# seeds = [30,42,50,70,80,100]
seeds = [30,42,50]
epochs = 40
time = 30 * 1000 # milliseconds
n_robots = 60
ls_cross_rate = [1]
ls_mut_rate = [0.8]
world = "bioinspired/src/world_map/maps/follow_token_map.txt"
for seed in seeds:
    np.random.seed(seed)
    for cross_rate in ls_cross_rate:
        for mut_rate in ls_mut_rate:
            GA = GeneticAlgorithmRunner(
                world=world,
                n_robots=n_robots,
                epochs=epochs,
                run_time=time,
                cross_rate=cross_rate,
                mut_rate=mut_rate,
                robot_type=TankNeuroRobot,
                seed=seed
            )
            GA.run()




