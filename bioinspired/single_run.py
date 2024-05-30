import pickle
from src.world_map.txt_to_map import WorldMap
from src.runners.genetic_algorithm import GeneticAlgorithmRunner
from src.utility.run_simulation import single_agent_run

map = r"bioinspired\src\world_map\maps\H_map.txt"
wm = WorldMap(skeleton_file=map, map_width=60, map_height=40, tile_size=15)
FILE = r"saved_runs\n_robots_250\epochs_20\epoch_time_90000\seed_30\1mut_1cross.pkl"
with open(FILE, "rb") as f:
    GA = pickle.load(f)
single_agent_run(wm,90000,GA.best_agent)
