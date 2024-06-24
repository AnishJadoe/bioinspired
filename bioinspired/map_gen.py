from src.world_map.map_generation import MapGenerator
from src.utility.constants import *

width = MAP_DIMS[0]
height = MAP_DIMS[1]
start_pos = START_POSITION  # Example start position (x, y)
num_tokens = N_TOKENS

map_gen = MapGenerator(width=width, height=height, start_pos=start_pos, num_tokens=num_tokens)
map_gen.generate_map()
map_gen.save_map_to_file("bioinspired/src/world_map/maps/random_map_1.txt")