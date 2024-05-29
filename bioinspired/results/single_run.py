import pickle
from .map_generator.txt_to_map import WorldMap
from .runner.genetic_algorithm import GeneticAlgorithmRunner
from .utility.run_simulation import single_agent_run

wm = WorldMap(skeleton_file="Maps/H_map.txt", map_width=60, map_height=40, tile_size=15)
with open("saved_runs/150pop_0.9mut_1cross_9000sim_50epochs.pkl", "rb") as f:
    GA = pickle.load(f)
single_agent_run(wm,90000,GA.best_agent)
