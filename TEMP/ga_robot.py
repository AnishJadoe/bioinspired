import numpy as np
from wheeled_robot_classv2 import Robot, Walls, Envir
import pygame

chromosome = np.array([[-25., -114.],
                       [-101., -101.],
                       [-212., -114.],
                       [-101., -101.],
                       [-101., -162.],
                       [-101., -101.],
                       [-162., -101.],
                       [-212., 66.],
                       [-101., 144.]],
                      )

time = 30000
# Simulation loop
pygame.init()

# Initialize the robot
start = (300, 200)
img_path = "/Users/anishjadoenathmisier/Documents/GitHub/BioInspiredIntelligence/robot.png"

robot = (Robot(start, img_path, 20, chromosome))
dims = (1200, 800)
env = Envir(dims)
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
