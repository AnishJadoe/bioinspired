import pickle
from bioinspired.src.world_map.world_map import WorldMap
from src.utility.run_simulation import single_agent_run
from src.utility.functions import load_GAs

FOLDER = r"saved_runs\n_robots_70\epochs_20"
wm = WorldMap(skeleton_file="bioinspired/src/world_map/maps/follow_token_map.txt", map_width=60, map_height=40, tile_size=15)

GAs = []
GAs = load_GAs(GAs,FOLDER)
all_fitness = [GA.best_eval for GA in GAs]
best_fitness_index = all_fitness.index(max(all_fitness))
best_run = GAs[best_fitness_index]

single_agent_run(wm,300000,best_run.best_agent)
