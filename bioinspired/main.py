"""
By running this file, the whole algorithm is run and plots
are made of the given paremeters
"""
# Loading in functions
import numpy as np
from src.runners.genetic_algorithm import GeneticAlgorithmRunner

seed = 30
np.random.seed(seed)
FOLDER = f"saved_runs/{seed}"
epochs = 10
time = 60000
n_robots = 20
ls_cross_rate = [1]
ls_mut_rate = [1,0.9,0.8]

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
        GA.save_run(folder=FOLDER, 
                    run_name=f"{n_robots}pop_{mut_rate}mut_{cross_rate}cross_{time}sim_{epochs}.pkl")




