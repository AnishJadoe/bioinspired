from typing import List
from src.robots.robots import BaseManualRobot, BaseRobot, TankNeuroRobot
from src.utility.functions import find_nearby_obstacles
from src.world_map.draw_functions import draw_end_pos, draw_gen, draw_line_to_next_token, draw_motor_speed, draw_next_token, draw_robot, draw_sensor_activation, draw_sensor_orientation, draw_tank_percentage, draw_time, draw_visited_tiles
from ..world_map.world_map import WorldMap
from ..robots.robot import Robot
import pygame
import pstats

import cProfile
from ..utility.constants import *


def profile_update_sensors_to_file(robot, nearby_obstacles, world_map):
    file_name = "run_simulation.prof"
    profiler = cProfile.Profile()
    profiler.enable()
    
    robot.update_sensors(nearby_obstacles, world_map)
    
    profiler.disable()
    profiler.dump_stats(file_name)
    return robot

    
# MAIN GAME LOOP
def run_simulation(wm: WorldMap, time,robot_type,  pop, n_robots, gen):
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

    wm.build_map()
    print(f"Walls in map: {len(wm.walls)}")
    clock = pygame.time.Clock()
    robots = list()
    pygame.init()
    for i in range(n_robots):
        if i <= 0.1*n_robots:
            # Best 10% are elite
            special = True
        else:
            special = False
            
        robots.append(
            robot_type(robot_id=i, startpos=wm.start_pos,targets=wm.tokens, end_target=wm.end_pos, 
                  chromosome=pop[i], special=special)
        )

    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    print(f"Loading took: {loadtime/1000} seconds")
    running = True
    dt = 0.01
    tokens_collected = []
    all_tanks_empty = False
    visited_cells = set()
    # Simulation loop
    while (not all_tanks_empty and running):
        clock.tick(30)
        timestamp = (pygame.time.get_ticks()/1000)
        
        if ((pygame.time.get_ticks() - loadtime)/1000) >= time:
            running = False
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # tanks_empty = [robot.tank_empty for robot in robots]
        # if all(tanks_empty):
        #     all_tanks_empty = True
        #     continue
        
        wm.update_map()
        for robot in robots:
            if not robot.mission_complete:
                robot.handler(world_map=wm, dt=dt, time=timestamp)
                token_to_collect = robot.current_target
                # if token_to_collect not in tokens_collected:
                #     tokens_collected.append(token_to_collect)
            elif robot.mission_complete:
                print(f"Robot {robot.id} finished")
                running = False
            # for cell in robot.visited_cells:
            #     visited_cells.add(cell)
            draw_robot(robot,wm.surf)
            if token_to_collect:
                draw_line_to_next_token(robot,token_to_collect,wm.surf)
                
        # if token_to_collect:
        #     draw_next_token(tokens_collected[-1], wm.surf)
        # else:
        #     draw_end_pos(wm.end_pos,wm.surf)
        
        # best_robot = 0
        # best_cells_visited = 0
        # for idx,robot in enumerate(robots):
        #     if robot.visited_cells:
        #         if len(robot.visited_cells) > best_cells_visited:
        #             best_cells_visited = len(robot.visited_cells)
        #             best_robot = idx
                    
        # draw_visited_tiles(visited_cells,wm.spatial_grid,wm.surf, GRAY)
        # draw_visited_tiles(robots[best_robot].visited_cells,wm.spatial_grid,wm.surf)
            
            
        draw_time(wm.surf, ((pygame.time.get_ticks() - loadtime)/1000))
        draw_gen(wm.surf,gen)
        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()
    wm.clear_map()
    return robots

def single_agent_run(wm: WorldMap, time, robot_type, chromosome):
    clock = pygame.time.Clock()
    pygame.init()
    wm.build_map()

    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    print(f"Loading took: {loadtime/1000} seconds")
    robot = robot_type((wm.start_pos.x, wm.start_pos.y), end_target=wm.end_pos,
                  chromosome=chromosome, targets=wm.tokens, robot_id=1)
    running = True
    dt = 0.01
    # Simulation loop
    while pygame.time.get_ticks() <= (time+loadtime) and running:
        if robot.reached_end:
            running = False
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        timestamp = (pygame.time.get_ticks()/1000)
        # Update frame by redrawing everything
        wm.update_map()
        nearby_obstacles = find_nearby_obstacles(robot,wm)
        robot.handler(nearby_obstacles=nearby_obstacles, dt=dt, time=timestamp)
        token_to_collect = robot.current_target
        
        draw_robot(robot,wm.surf)
        if token_to_collect:
            draw_next_token(token_to_collect, wm.surf)
        else:
            draw_end_pos(wm.end_pos,wm.surf)
            
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
    dt = 0.01
    tokens_collected = []
    # Simulation loop
    while pygame.time.get_ticks() <= (time+loadtime) and running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


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

def manual_mode(wm: WorldMap, robot:BaseRobot, drawfunc):
    clock = pygame.time.Clock()
    
    pygame.init()
    wm.build_map()

    dt = 0.01
    lasttime = pygame.time.get_ticks()
    loadtime = lasttime
    running = True
    robot = robot(startpos=(wm.start_pos.x, wm.start_pos.y),targets=wm.tokens, end_target=wm.end_pos)
    # Simulation loop
    while running:
        if robot.mission_complete:
            print(f"Congrats you won within {(pygame.time.get_ticks()- loadtime)/1000} seconds")
            running=False
        clock.tick(30)
        wm.update_map()

        
        nearby_obstacles = find_nearby_obstacles(robot,wm)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            robot.handler(nearby_obstacles=nearby_obstacles,dt=dt,event=event)  
        robot.handler(nearby_obstacles=nearby_obstacles, dt=dt)
    
        
        draw_robot(robot,wm.surf)
        draw_time(wm.surf, (pygame.time.get_ticks()- loadtime)/1000)
        drawfunc(robot,wm)

        
        lasttime = pygame.time.get_ticks()
        pygame.display.update()

    pygame.quit()