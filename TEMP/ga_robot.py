import numpy as np
from Code.robot import Robo
from Code.constants import * 
import pygame

chromosome = np.array([[ 23 , 11],
 [ 43,  44],
 [-48,   4],
 [-42, -12],
 [-25, -17],
 [-22,   3],
 [ 35, -32],
 [-17, -34],
 [-46,   1],
 [-24, -48],
 [-23, -16],
 [-15, -35],
 [-43, -30],
 [ 17,  31],
 [ 41,  -9],
 [ 10, -29],
 [-30,  43],
 [ 24, -44],
 [ 44,  23],
 [ 45, -38],
 [-21, -27],
 [  0, -30],
 [ 32, -22],
 [-19, -21],
 [-22,  38],
 [ 37, -22],
 [-41, -33],
 [-17, -34],
 [ 23, -34],
 [ 33,  37],
 [ 18,  43],
 [-45, -11]],
                      )

time = 30000
# Simulation loop
pygame.init()

# Initialize the robot
img_path = "/Users/anishjadoenathmisier/Documents/GitHub/BioInspiredIntelligence/robot.png"

robot = (Robot(start, img_path, 20, chromosome))
env = Envir(WORLD)
wall = Walls(20, dims)

env.map.fill((255, 255, 255))
dt = 0
lasttime = pygame.time.get_ticks()

while pygame.time.get_ticks() <= time:

    robot.get_collision(wall.obstacles)
    robot.get_sensor(wall.obstacles, env.map)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()

        robot.move(env.height, env.width, dt, event, auto=True)

    dt = (pygame.time.get_ticks() - lasttime) / 1000
    lasttime = pygame.time.get_ticks()

    robot.move(env.height, env.width, dt, auto=True)

    pygame.display.update()

    env.map.fill((255, 255, 255))
    wall.draw(env.map)
    robot.draw(env.map)

pygame.quit()

print(robot.dist_travelled)
