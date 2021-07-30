import numpy as np
from wheeled_robot_class import Robot, Walls, Envir
import pygame

chromosome = np.array([[0.02497166, 0.44903877, 0.23747856, 0.04961437],
                       [0.72459829, 0.111593, 0.6083195, 0.28104975],
                       [0.17349605, 0.37975152, 0.80144578, 0.39209358],
                       [0.75076965, 0.1254863, 0.77277142, 0.23693332],
                       [0.67739313, 0.56578155, 0.92914472, 0.38732252],
                       [0.06616519, 0.01915549, 0.82685856, 0.52473564]])

time = 15000
# Simulation loop
pygame.init()

# Initialize the robot
start = (300, 200)
img_path = "/Users/anishjadoenathmisier/Documents/GitHub/BioInspiredIntelligence/robot.png"

robot = (Robot(start, img_path, 25, chromosome))
dims = (1400, 1000)
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
