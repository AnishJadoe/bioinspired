from map_generator.txt_to_map import WorldMap
from robot import Robot
import pygame
import numpy as np
import os
from constants import *

# Initialize the robot
script_dir = os.path.dirname(__file__)
rel_path = "../Images/robot.png"
img_path = os.path.join(script_dir, rel_path)


# MAIN GAME LOOP
def run_simulation(wm: WorldMap, time, pop, n_robots):
    """This is the main game loop of the algorithm, it is called by the Genetic Algorithm class in
    the main loop. It yields the scores of the corresponding individual chromosomes and saves the result of each run to the genetic algorithm class

    Args:
        time (int): The runtime of each epoch
        pop (numpy array): All the individual robots part of this generation
        n_robots (int): The amount of robots to be used in the algorithm
        GA (Genetic Algorithm class): This is the class that contains all the genetic algorithm functions

    Returns:
        scores (float)]: The scores given to each individual
    """
    global img_path

    clock = pygame.time.Clock()
    ls_robots = list()
    
    simulation_results = {}
    fitness = list()
    tot_abs_dist = list()
    tot_rel_dist = list()
    tot_coll = list()
    tot_token = list()
    
    pygame.init()
    wm.build_map()

    for i in range(n_robots):
        ls_robots.append(
            Robot((wm.start_pos.x, wm.start_pos.y), width=15, chromosome=pop[i])
        )

    dt = 0
    lasttime = pygame.time.get_ticks()

    # Simulation loop
    while pygame.time.get_ticks() <= time:
        clock.tick(60)
        dt = (pygame.time.get_ticks() - lasttime) / 1000

        # Update frame by redrawing everything
        wm.update_map(ls_robots)
        for robot in ls_robots:
            FOV = robot.find_obstacles(wm)
            robot.get_sensor_FOV(FOV)
            wall_collided = robot.get_collision(wm)
            robot.move(wall_collided, dt, auto=True)
            robot.draw(wm.surf)

        print(f"Time Elapsed {pygame.time.get_ticks() / 1000}")
        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()

    for robot in ls_robots:
        fitness.append(robot.get_reward())
        tot_rel_dist.append(robot.avg_dist)
        tot_abs_dist.append(robot.dist_travelled)
        tot_coll.append(robot.collision)
        tot_token.append(robot.token)

    simulation_results["pop_fitness"] = fitness
    simulation_results["pop_rel_distance"] = tot_rel_dist
    simulation_results["pop_abs_distance"] = tot_abs_dist
    simulation_results["pop_collisions"] = tot_coll
    simulation_results["pop_token"] = tot_token
    
    return simulation_results
