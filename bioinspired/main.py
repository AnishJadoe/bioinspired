"""
By running this file, the whole algorithm is run and plots
are made of the given paremeters
"""
# Loading in functions
import numpy as np
from src.runners.genetic_algorithm import GeneticAlgorithmRunner
from src.utility.constants import *
# seeds = [30,42,50,70,80,100]
seeds = [30,42,50]
epochs = 20
time = 30 * 1000 # milliseconds
n_robots = 70
ls_cross_rate = [1]
ls_mut_rate = [0.9,0.85,0.8]

for seed in seeds:
    np.random.seed(seed)
    for cross_rate in ls_cross_rate:
        for mut_rate in ls_mut_rate:
            GA = GeneticAlgorithmRunner(
                n_robots=n_robots,
                epochs=epochs,
                run_time=time,
                cross_rate=cross_rate,
                mut_rate=mut_rate,
                seed=seed
            )
            GA.run()




