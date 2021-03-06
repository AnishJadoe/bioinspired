from walls import Walls
from robot import Robot
from environment import Envir
from draw import Draw
import pygame
import numpy as np
import os

# Initialize the robot
start = (300, 200)

script_dir = os.path.dirname(__file__)
rel_path = "../Images/robot.png"
img_path = os.path.join(script_dir, rel_path)

np.random.RandomState()

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

    global start
    global img_path

    clock = pygame.time.Clock()
    ls_robots = list()
    scores = list()

    tot_abs_dist = list()
    tot_avg = list()
    tot_coll = list()
    tot_reward = list()
    tot_flips = list()
    tot_token = list()

    dims = (1400, 800)

    environment = Envir(dims)
    environment.map.fill((255, 255, 255))

    # Obtain the walls
    wall = Walls(20, dims)

    draw = Draw(dims)

    for i in range(n_robots):
        ls_robots.append(Robot(start, img_path, 15, pop[i]))

    dt = 0
    lasttime = pygame.time.get_ticks()
    # Simulation loop
    while pygame.time.get_ticks() <= time:
        clock.tick(120)

        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:
                if event.key == 48:
                    robot.x = 200
                    robot.y = 200

            if event.type == pygame.QUIT:
                pygame.quit()

            for robot in ls_robots:
                robot.move(environment.height,
                           environment.width,
                           dt,
                           event,
                           auto=True)

        dt = (pygame.time.get_ticks() - lasttime) / 1000
        lasttime = pygame.time.get_ticks()

        wall.get_tokens() # Populate the world with rewards 
        environment.map.fill((255, 255, 255))
        wall.draw(environment.map, ls_robots)

        for robot in ls_robots:
            robot.get_sensor(wall.obstacles, environment.map) 
            if sum(robot.sensor[1:]) > 0:
                robot.get_collision(wall.obstacles)
            robot.move(environment.height, environment.width, dt, auto=True)

            robot.draw(environment.map)

        draw.write_info(gen=GA.gen,
                        time=pygame.time.get_ticks() / 1000,
                        map=environment.map)

        pygame.display.update()

    pygame.quit()

    for robot in ls_robots:
        scores.append(robot.get_reward())

    for robot in ls_robots:
        tot_avg.append(robot.avg_dist)
        tot_abs_dist.append(robot.dist_travelled)
        tot_coll.append(robot.collision)
        tot_reward.append(robot.get_reward())
        tot_flips.append(robot.flip)
        tot_token.append(robot.token)

    GA.reward_gen.append(np.mean(tot_reward))
    GA.dist_gen.append(np.mean(tot_abs_dist))
    GA.rel_dist_gen.append(np.mean(tot_avg))
    GA.coll_gen.append(np.mean(tot_coll))
    GA.token_gen.append(np.mean(tot_token))

    return scores
