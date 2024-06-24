from src.utility.functions import  profile_func, get_init_chromosomes_NN
from src.world_map.world_map import WorldMap
from src.utility.constants import *
from src.robots.robots import SearchingTankNeuroRobot, TankNeuroRobot

@profile_func
def mock_test(robot):
    robot.handler()
    return
    
wm = WorldMap(skeleton_file="bioinspired/src/world_map/maps/random_map_1.txt")
wm.build_map()
chromosome = get_init_chromosomes_NN(1, N_INPUTS,N_OUTPUTS,N_HIDDEN)[0]
robot = SearchingTankNeuroRobot(wm.start_pos,wm.tokens,None,1,chromosome)

mock_test(robot)