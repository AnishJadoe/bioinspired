from src.world_map.draw_functions import *
from src.robots.robots import BaseManualRobot, DebugBot
from src.world_map.world_map import WorldMap
from src.utility.run_simulation import manual_mode

def draw_func(robot:BaseRobot,wm:WorldMap):
    draw_next_token(robot.current_target,wm.surf)
    draw_end_pos(wm.end_pos,wm.surf)
    draw_motor_speed(robot,wm.surf)
    # draw_sensor_orientation(robot,wm.surf)
    # draw_sensor_activation(robot,wm.surf)
    # debug_theta(robot,wm.surf)

    
map = r"bioinspired/src/world_map/maps/follow_token_map.txt"
wm = WorldMap(skeleton_file=map, map_width=60, map_height=40, tile_size=15)


manual_mode(wm, DebugBot, draw_func)
