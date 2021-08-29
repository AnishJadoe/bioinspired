import numpy as np
import pygame
import math
from functions import calc_distance
from functions import calc_angle


class Robot:
    def __init__(self, startpos, robot_img, width, chromosome):

        self.m2p = 3779.52  # meters 2 pixels
        self.w = width * 10
        self.init_pos = startpos
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0
        self.vl = 1
        self.vr = 1
        self.maxspeed = 0.01 * self.m2p
        self.minspeed = -0.01 * self.m2p

        self.ls_tot_speed = list()

        self.ls_theta = []
        self.ls_x = []
        self.ls_y = []
        self.omega = 0

        self.dist_travelled = 0
        self.avg_dist = 0
        self.flip = 0
        self.stuck = 0

        self.sensor = [
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # 1 tactile sensor (sensor[0]) and 5 ultrasound

        self.width = width
        self.length = width
        self.collision = 0
        self.reward = 0

        self.img = pygame.image.load(robot_img)
        self.img = pygame.transform.scale(self.img, (self.width, self.length))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        self.chromosome = chromosome

    def draw(self, world):
        world.blit(self.rotated, self.rect)

    def get_collision(self, obstacles):

        self.sensor[0] = 0

        for obstacle in obstacles:
            if self.rect.colliderect(obstacle):
                self.collision += 1
                self.sensor[0] = 1

    def get_reward(self, rewards):

        for reward in rewards:
            if self.rect.colliderect(reward):
                self.reward += 1
                rewards.pop(reward)

    def move(self, height, width, dt, event=None, auto=False):

        self.ls_x.append(self.x)
        self.ls_y.append(self.y)
        self.ls_theta.append(self.theta)

        if self.x == self.ls_x[-1] and self.y == self.ls_y[-1]:
            self.stuck += 1

        self.ls_tot_speed.append(self.vl + self.vr)

        if auto:

            states = np.zeros(shape=(1, 10), dtype=object)
            states[:, 0:-4] = self.sensor
            # Give the robot its own state as input
            states[:, -4] = self.theta
            states[:, -3] = self.omega
            states[:, -2] = self.vl
            states[:, -1] = self.vr

            actions = np.dot(self.chromosome.T, states.T)

            # Instead of having 2 options, the robot now has 4
            # such that it is also able to brake

            self.vl += actions[0] * 0.015
            self.vr += actions[1] * 0.015

        else:
            if event is not None:
                if event.type == pygame.KEYDOWN:

                    if event.key == 49:  # 1 is accelerate left
                        self.vl += 0.01 * self.m2p

                    elif event.key == 51:  # 3 is decelerate left
                        self.vl -= 0.01 * self.m2p

                    elif event.key == 54:  # 8 is accelerate right
                        self.vr += 0.01 * self.m2p

                    elif event.key == 56:  # 0 is decelerate right
                        self.vr -= 0.01 * self.m2p

                # check to see if the rotational velocity of the robot is not exceding
        # the maximum rotational velocity

        self.omega = (self.vr - self.vl) / self.w

        if self.omega >= 0.02 * math.pi:
            self.flip += 1
            self.omega = 0.02 * math.pi

        self.theta += self.omega * dt

        if self.theta > 2 * math.pi or self.theta < -2 * math.pi:
            self.theta = 0

        if self.sensor[0]:
            # Whenever there is a collision we move in the opposite direction
            # but also dissipate some energy, hence the -0.1
            self.x += (-0.1 * (self.vr + self.vl) * 0.5 * math.cos(self.theta) * dt)[0]
            self.y -= (-0.1 * (self.vr + self.vl) * 0.5 * math.sin(self.theta) * dt)[0]

        else:

            self.x += (((self.vl + self.vr) / 2) * math.cos(self.theta) * dt)[0]
            self.y -= (((self.vl + self.vr) / 2) * math.sin(self.theta) * dt)[0]

        # Detects if we're within map borders

        if self.x < 0 + 0.5 * self.length:
            self.x = 0 + 0.5 * self.length

        if self.x >= height - 1.2 * self.length:
            self.x = height - 1.2 * self.length

        if self.y < 0 + 0.5 * self.width:
            self.y = 0 + 0.5 * self.width

        if self.y >= width - 1.2 * self.width:
            self.y = width - 1.2 * self.width

        # set min speed
        self.vr = max(self.vr, self.minspeed)
        self.vl = max(self.vl, self.minspeed)

        # set max speed
        self.vr = min(self.vr, self.maxspeed)
        self.vl = min(self.vl, self.maxspeed)

        self.rotated = pygame.transform.rotozoom(self.img, math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        return

    def get_sensor(self, obstacles, map):

        if self.sensor[0] == 0:

            self.sensor = [0, 0, 0, 0, 0, 0]

            for obstacle in obstacles:
                dist_to_wall = calc_distance((self.x, self.y), (obstacle.x, obstacle.y))

                if dist_to_wall <= 75:
                    angle_w_wall = calc_angle(
                        (self.x, self.y), (obstacle.x, obstacle.y)
                    )

                    if 0 <= angle_w_wall < (math.pi * 2 * 1 / 5):
                        pygame.draw.line(
                            map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y)
                        )
                        self.sensor[1] = 1 * 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 1 / 5) <= angle_w_wall < (math.pi * 2 * 2 / 5):
                        pygame.draw.line(
                            map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y)
                        )
                        self.sensor[2] = 1 * 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 2 / 5) <= angle_w_wall < (math.pi * 2 * 3 / 5):
                        pygame.draw.line(
                            map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y)
                        )
                        self.sensor[3] = 1 * 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 3 / 5) <= angle_w_wall < (math.pi * 2 * 4 / 5):
                        pygame.draw.line(
                            map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y)
                        )
                        self.sensor[4] = 1 * 1 / max(1, dist_to_wall)

                    else:
                        pygame.draw.line(
                            map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y)
                        )
                        self.sensor[5] = 1 * 1 / max(1, dist_to_wall)

    def get_reward(self, time):

        for i in range(1, len(self.ls_x)):
            # Take the line integral

            delta_x = float(self.ls_x[i] - self.ls_x[i - 1])
            delta_y = float(self.ls_y[i] - self.ls_y[i - 1])
            delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)
            self.dist_travelled += delta_r

        self.avg_dist = math.sqrt(
            (self.init_pos[0] - self.x) ** 2 + (self.init_pos[1] - self.y) ** 2
        )

        score = (
            self.dist_travelled * 0.5
            + self.avg_dist * 2
            - self.collision * 1.5
            + self.reward * 5
        )

        fitness = score / (
            self.dist_travelled + self.avg_dist + self.collision + self.reward
        )

        return float(fitness)