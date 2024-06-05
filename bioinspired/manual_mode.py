import pickle
from src.world_map.txt_to_map import WorldMap
from src.utility.run_simulation import manual_mode

map = r"bioinspired\src\world_map\maps\test_map.txt"
wm = WorldMap(skeleton_file=map, map_width=60, map_height=40, tile_size=15)

manual_mode(wm)
