from ..world_map.txt_to_map import WorldMap
from ..robot.robot import Robot
import pygame
import numpy as np
import os
from ..utility.constants import *

# Initialize the robot
script_dir = os.path.dirname(__file__)
rel_path = "../Images/robot.png"
img_path = os.path.join(script_dir, rel_path)

def draw_time(world, time):
    font = pygame.font.SysFont(None, 36)
    txt = font.render(f'Time: {round(time/1000,1)}', True, (0,0,0))
    world.blit(txt,(750,50))

def draw_gen(world, gen):
    font = pygame.font.SysFont(None, 24)
    txt = font.render(f'Gen: {gen}', True, (0,0,0))
    world.blit(txt,(750,75))

# MAIN GAME LOOP
def run_simulation(wm: WorldMap, time, pop, n_robots, gen):
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
    clock = pygame.time.Clock()
    ls_robots = list()
    
    simulation_results = {}
    fitness = list()
    tot_abs_dist = list()
    tot_rel_dist = list()
    tot_coll = list()
    tot_token = list()
    wm.build_map()
    # print(f"Walls in map: {len(wm.walls)}")
    pygame.init()
    for i in range(n_robots):
        if i <= 0.1*n_robots:
            # Best 10% are elite
            special = True
        else:
            special = False
        ls_robots.append(
            Robot(robot_id=i, startpos=(wm.start_pos.x, wm.start_pos.y), width=20, 
                  chromosome=pop[i], token_locations=wm.tokens.copy(), special_flag=special)
        )

    dt = 0
    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    print(f"Loading took: {loadtime/1000} seconds")
    # Simulation loop
    while pygame.time.get_ticks() <= (time+loadtime):
        clock.tick(20)
        dt = (pygame.time.get_ticks() - lasttime) / 1000

        # Update frame by redrawing everything
        wm.update_map()
        for robot in ls_robots:
            robot.find_obstacles(wm)
            robot.get_tokens()
            robot.move(robot.get_collision(wm), dt, auto=True)
            robot.draw(wm.surf)

        draw_time(wm.surf, (pygame.time.get_ticks()- loadtime))
        draw_gen(wm.surf,gen)
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

def single_agent_run(wm: WorldMap, time, chromosome):
    clock = pygame.time.Clock()
    
    pygame.init()
    wm.build_map()

    dt = 0
    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    print(f"Loading took: {loadtime/1000} seconds")
    agent = Robot((wm.start_pos.x, wm.start_pos.y), width=20, 
                  chromosome=chromosome, token_locations=wm.tokens.copy(), special_flag=True)
    # Simulation loop
    while pygame.time.get_ticks() <= (time+loadtime):
        clock.tick(60)
        dt = (pygame.time.get_ticks() - lasttime) / 1000
        # Update frame by redrawing everything
        wm.update_map()
        agent.find_obstacles(wm)
        agent.get_tokens()
        agent.move(agent.get_collision(wm), dt, auto=True)
        agent.draw(wm.surf)

        draw_time(wm.surf, (pygame.time.get_ticks()- loadtime))
        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()
