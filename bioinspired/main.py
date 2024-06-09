"""
By running this file, the whole algorithm is run and plots
are made of the given paremeters
"""
# Loading in functions
import numpy as np
from src.runners.genetic_algorithm import GeneticAlgorithmRunner

# seeds = [30,42,50,70,80,100]
seeds = [30]
epochs = 30
time = 30 * 1000 # milliseconds
n_robots = 200
ls_cross_rate = [1]
ls_mut_rate = [0.8] 

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
            )
            GA.run()
            FOLDER = f"saved_runs/n_robots_{n_robots}/epochs_{epochs}/epoch_time_{time}/seed_{seed}"
            GA.save_run(folder=FOLDER, 
                        run_name=f"{mut_rate}mut_{cross_rate}cross.pkl")




