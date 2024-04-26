"""
By running this file, the whole algorithm is run and plots
are made of the given paremeters
"""
# Loading in functions
import numpy as np
from plot import plot_mean_results, plot_top_20_results
from genetic_algorithm import GeneticAlgorithmRunner
from map_generator.txt_to_map import WorldMap

np.random.seed(30)

epochs = 1
time = 60000
n_robots = 20
ls_cross_rate = [1]
ls_mut_rate = [1]

wm = WorldMap(skeleton_file="Maps/H_map.txt", map_width=60, map_height=40, tile_size=20)
GA = GeneticAlgorithmRunner(
    world_map=wm,
    n_robots=n_robots,
    epochs=epochs,
    run_time=time,
    cross_rate=1,
    mut_rate=1,
)
GA.run()


# GA.save_run("saved_runs/test_run_4")

# plot_mean_results(GA)
# plot_top_20_results(GA)
