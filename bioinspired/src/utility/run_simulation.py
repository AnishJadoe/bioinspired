from ..world_map.txt_to_map import WorldMap
from ..robot.robot import Robot
import pygame

import cProfile
import pstats
import io
from ..utility.constants import *


def profile_update_sensors_to_file(robot, nearby_obstacles, world_map):
    file_name = "update_sensors.prof"
    profiler = cProfile.Profile()
    profiler.enable()
    
    robot.update_sensors(nearby_obstacles, world_map)
    
    profiler.disable()
    profiler.dump_stats(file_name)
    return robot

BLUE = (0,0,255)
def draw_time(world, time):
    font = pygame.font.SysFont(None, 36)
    txt = font.render(f'Time: {round(time,1)}', True, BLUE)
    world.blit(txt,(750,50))

def draw_gen(world, gen):
    font = pygame.font.SysFont(None, 24)
    txt = font.render(f'Gen: {gen}', True, BLUE)
    world.blit(txt,(800,75))

def draw_next_token(world, token):
    pygame.draw.rect(world,BLUE, token)

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
    tot_cells_explored = list()
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
            Robot(robot_id=i, startpos=(wm.start_pos.x, wm.start_pos.y), endpos=wm.end_pos, width=20, 
                  chromosome=pop[i], token_locations=wm.tokens.copy(), special_flag=special)
        )

    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    print(f"Loading took: {loadtime/1000} seconds")
    running = True
    dt = 0
    tokens_collected = []
    all_tanks_empty = False
    robots_finished = set()
    # Simulation loop
    while (not all_tanks_empty and running):
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        timestamp = (pygame.time.get_ticks()/1000)
        dt = (pygame.time.get_ticks() - lasttime) / 1000
        tanks_empty = [robot.tank_empty for robot in ls_robots]
        if all(tanks_empty):
            all_tanks_empty = True
            continue
        # Update frame by redrawing everything
        wm.update_map()
        for robot in ls_robots:
            if not robot.reached_end:
                robot.update_state(timestamp)
                nearby_obstacles = robot.find_position(wm)
                robot.update_sensors(nearby_obstacles,wm)
                token_to_collect = robot.get_tokens(timestamp)
                if token_to_collect not in tokens_collected:
                    tokens_collected.append(token_to_collect)
            elif robot.reached_end:
                robots_finished.add(robot.id)
                running = False

            robot.move(robot.get_collision(nearby_obstacles), dt, auto=True)
            robot.draw(wm.surf)

        if token_to_collect:
            draw_next_token(wm.surf, tokens_collected[-1])
        else:
            draw_next_token(wm.surf,wm.end_pos)
        draw_time(wm.surf, (pygame.time.get_ticks()/1000))
        draw_gen(wm.surf,gen)
        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()
    for id in robots_finished:
        print(f"Robot {id} finished the course")
    for robot in ls_robots:
        fitness.append(robot.get_reward())
        tot_cells_explored.append(robot.avg_dist)
        tot_abs_dist.append(robot.dist_travelled)
        tot_coll.append(robot.collision)
        tot_token.append(robot.token)
        tot_cells_explored.append(len(robot.visited_cells))

    simulation_results["pop_fitness"] = fitness
    simulation_results["pop_rel_distance"] = tot_rel_dist
    simulation_results["pop_abs_distance"] = tot_abs_dist
    simulation_results["pop_collisions"] = tot_coll
    simulation_results["pop_token"] = tot_token
    simulation_results["pop_cells_explored"] = tot_cells_explored

    return simulation_results

def single_agent_run(wm: WorldMap, time, chromosome):
    clock = pygame.time.Clock()
    pygame.init()
    wm.build_map()

    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    print(f"Loading took: {loadtime/1000} seconds")
    robot = Robot((wm.start_pos.x, wm.start_pos.y), endpos=wm.end_pos,width=20,
                  chromosome=chromosome, token_locations=wm.tokens, special_flag=True)
    running = True
    dt = 0
    # Simulation loop
    while pygame.time.get_ticks() <= (time+loadtime) and running:
        if robot.reached_end:
            running = False
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update frame by redrawing everything
        dt = (pygame.time.get_ticks() - lasttime) / 1000
        wm.update_map()
        nearby_obstacles = robot.find_position(wm)
        robot.update_state()
        robot.update_sensors(nearby_obstacles,wm)
        token_to_collect = robot.get_tokens(0)
        robot.move(robot.get_collision(nearby_obstacles), dt, auto=True)
        robot.draw(wm.surf)
        
        draw_next_token(wm.surf, token_to_collect)


        draw_time(wm.surf, (pygame.time.get_ticks()/1000))
        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()

def multi_agent_run(wm: WorldMap, time, chromosomes):
    clock = pygame.time.Clock()
    pygame.init()
    wm.build_map()

    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    print(f"Loading took: {loadtime/1000} seconds")
    ls_robots = list()
    for chromosome in chromosomes:
        ls_robots.append(Robot((wm.start_pos.x, wm.start_pos.y),endpos=wm.end_pos, width=20, 
                    chromosome=chromosome, token_locations=wm.tokens.copy(), special_flag=True))
    running = True
    dt = 0
    tokens_collected = []
    # Simulation loop
    while pygame.time.get_ticks() <= (time+loadtime) and running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update frame by redrawing everything
        dt = (pygame.time.get_ticks() - lasttime) / 1000
        timestamp = pygame.time.get_ticks()/1000
        wm.update_map()
        for robot in ls_robots:
            robot.update_state()
            nearby_obstacles = robot.find_position(wm)
            robot.update_sensors(nearby_obstacles,wm)
            token_to_collect = robot.get_tokens(timestamp)
            if token_to_collect not in tokens_collected:
                tokens_collected.append(token_to_collect)
            if robot.found_all_tokens:
                robot.get_end_tile()
            robot.move(robot.get_collision(nearby_obstacles), dt, auto=True)
            robot.draw(wm.surf)

        draw_next_token(wm.surf, token_to_collect)


        draw_time(wm.surf, (pygame.time.get_ticks()/1000))
        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()

def manual_mode(wm: WorldMap):
    clock = pygame.time.Clock()
    
    pygame.init()
    wm.build_map()

    dt = 0
    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    running = True
    robot = Robot((wm.start_pos.x, wm.start_pos.y), width=30, 
                  chromosome=[], token_locations=wm.tokens, special_flag=True)
    # Simulation loop
    while running:

        wm.update_map()
        nearby_obstacles = robot.find_position(wm)
        robot.update_sensors(nearby_obstacles,wm)
        # robot.get_tokens()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            robot.move(robot.get_collision(nearby_obstacles),dt,event)
        robot.move(robot.get_collision(nearby_obstacles), dt)
        dt = 0.01 #(pygame.time.get_ticks() - lasttime) / 1000
        robot.draw(wm.surf)

        draw_time(wm.surf, (pygame.time.get_ticks()- loadtime))
        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()