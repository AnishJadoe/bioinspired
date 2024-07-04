"""
By running this file, the whole algorithm is run and plots
are made of the given paremeters
"""
# Loading in functions
import numpy as np
from src.world_map.world_map import WorldMap
from src.robots.robots import SearchingTankNeuroRobot, TankNeuroRobot, ForagingNeuroBot
from src.runners.genetic_algorithm import GeneticAlgorithmRunner
from src.utility.constants import *
# seeds = [30,42,50,70,80,100]
seeds = [30,42,50]
epochs = 75
time = 90 #s
n_robots = 5
ls_cross_rate = [1]
ls_mut_rate = [0.8]
world = "bioinspired/src/world_map/maps/random_map_small.txt"
wm = WorldMap(skeleton_file=world)
for seed in seeds:
    np.random.seed(seed)
    for cross_rate in ls_cross_rate:
        for mut_rate in ls_mut_rate:
            GA = GeneticAlgorithmRunner(
                world=wm,
                n_robots=n_robots,
                epochs=epochs,
                run_time=time,
                cross_rate=cross_rate,
                mut_rate=mut_rate,
                robot_type=ForagingNeuroBot,
                seed=seed
            )
            GA.run()




