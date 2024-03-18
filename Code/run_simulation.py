from walls import Map
from robot import Robot
from world import World
from draw import Draw
import pygame
import numpy as np
import os
from constants import *

# Initialize the robot
script_dir = os.path.dirname(__file__)
rel_path = "../Images/robot.png"
img_path = os.path.join(script_dir, rel_path)


# MAIN GAME LOOP
def run_simulation(time, pop, n_robots, GA):
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
    pygame.init()
    pygame.display.set_mode(size=(1400, 800), display=0)
    global img_path

    clock = pygame.time.Clock()
    ls_robots = list()
    scores = list()

    tot_abs_dist = list()
    tot_avg = list()
    tot_coll = list()
    tot_reward = list()
    tot_token = list()

    world = World(MAP_SIZE)
    world.draw_H_world()

    # Obtain the walls
    wall = Map(world=world.map, cell_size=20)
    draw = Draw(MAP_SIZE)

    for i in range(n_robots):
        ls_robots.append(Robot(START_POSITION, width=15, chromosome=pop[i]))

    dt = 0
    lasttime = pygame.time.get_ticks()

    # Simulation loop
    while pygame.time.get_ticks() <= time:
        clock.tick(20)
        dt = (pygame.time.get_ticks() - lasttime) / 1000

        # Update frame by redrawing everything
        world.map.fill((255, 255, 255))
        wall.draw_walls(world.map)
        wall.update_tokens(world.map, ls_robots)

        for robot in ls_robots:
            FOV = robot.find_obstacles(wall)

            robot.get_sensor_FOV(FOV)
            if sum(robot.sensor[1:]) > 0:
                robot.get_collision(FOV)
            robot.move(wall, dt, auto=True)
            robot.draw(world.map)

        draw.write_info(gen=GA.gen, time=pygame.time.get_ticks() / 1000, map=world.map)

        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()

    for robot in ls_robots:
        scores.append(robot.get_reward(wall))

    for robot in ls_robots:
        tot_avg.append(robot.avg_dist)
        tot_abs_dist.append(robot.dist_travelled)
        tot_coll.append(robot.collision)
        tot_reward.append(robot.get_reward(wall))
        tot_token.append(robot.token)

    GA.reward_gen.append(np.mean(tot_reward))
    GA.dist_gen.append(np.mean(tot_abs_dist))
    GA.rel_dist_gen.append(np.mean(tot_avg))
    GA.coll_gen.append(np.mean(tot_coll))
    GA.token_gen.append(np.mean(tot_token))

    return scores
