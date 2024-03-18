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
        self.current_pos = (self.x,self.y)
        self.theta = 0
        self.vl = 1
        self.vr = 1
        self.maxspeed = 300
        self.minspeed = -300
        self.all_states = np.array(np.meshgrid([0, 1],[0,1],[0,1],[0,1],[0,1])).T.reshape(-1, 5)

        self.ls_tot_speed = list()
        self.visited_cells = set()
        self.ls_theta = []
        self.ls_x = []
        self.ls_y = []
        self.omega = 0

        self.dist_travelled = 0
        self.avg_dist = 0


        self.sensor = [
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # 1 tactile sensor (sensor[0]) and 5 ultrasound

        self.width = width * 2
        self.length = width * 2
        self.collision = 0
        self.token = 1

        self.img = pygame.image.load(robot_img)
        self.img = pygame.transform.scale(self.img, (self.width, self.length))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        self.chromosome = chromosome

    def draw(self, world):
        '''
        Draws the robot on the world
        '''
        world.blit(self.rotated, self.rect)

    def get_collision(self, obstacles):
        '''
        Function to check if we have collided with an obstacle, the sensor
        value is changed whenever there is a collision
        '''

        self.sensor[0] = 0
        for obstacle in obstacles:
            if self.rect.colliderect(obstacle):
                self.collision += 1  # this is to keep track of the amount of collisions the robot has made
                self.sensor[0] = 1

    def get_token(self, tokens):
        '''
        Function to check if we have collided with a token, when this happens the
        robot obtains a token
        '''

        for token in tokens:
            if self.rect.colliderect(token):
                print(self.token)
                print(self.token)
                tokens.pop(token)

    def move(self, height, width, dt, event=None, auto=False):

        self.ls_x.append(self.x)
        self.ls_y.append(self.y)
        self.ls_theta.append(self.theta)

        # Calculating the amount of distance traveled since last time step and adding
        # this to absolute distance travelled
        if len(self.ls_x) >= 2:
            delta_x = float(self.ls_x[-2] - self.ls_x[-1])
            delta_y = float(self.ls_y[-2] - self.ls_y[-1])
            delta_r = np.sqrt(delta_x**2 + delta_y**2)
            self.dist_travelled += delta_r

        if self.x == self.ls_x[-1] and self.y == self.ls_y[-1]:
            self.stuck += 1

        self.ls_tot_speed.append(self.vl + self.vr)

        if auto:

            # states = np.zeros(shape=(1, 10), dtype=object)
            # states[:, 0:-4] = self.sensor
            # Give the robot its own state as input
            # states[:, -4] = self.theta
            # states[:, -3] = self.omega
            # states[:, -2] = self.vl
            # states[:, -1] = self.vr

            #states = np.array([self.sensor])
            #actions = np.dot(self.chromosome.T, states.T)

            # Instead of having 2 options, the robot now has 4
            # such that it is also able to brake
            
            index = np.where((self.all_states == self.sensor[1:]).all(axis=1))[0][0]
            actions = self.chromosome[index]
            self.vl += actions[0] 
            self.vr += actions[1] 

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

        if self.sensor[0]:
            # Whenever there is a collision we move in the opposite direction
            # but also dissipate some energy, hence the minus sign
            self.vr = -0.1 * self.vr
            self.vl = -0.1 * self.vl

        # set min speed
        self.vr = max(self.vr, self.minspeed)
        self.vl = max(self.vl, self.minspeed)

        # set max speed
        self.vr = min(self.vr, self.maxspeed)
        self.vl = min(self.vl, self.maxspeed)
            
        # check to see if the rotational velocity of the robot is not exceding
        # the maximum rotational velocity

        self.omega = (self.vr - self.vl) / self.w
        self.theta += self.omega * dt

        if self.theta > 2 * math.pi or self.theta < -2 * math.pi:
            self.theta = 0

        self.x += (((self.vl + self.vr) / 2) * math.cos(self.theta) *
                    dt)
        self.y -= (((self.vl + self.vr) / 2) * math.sin(self.theta) *
                    dt)


        # Detects if we're within map borders

        if self.x < 0 + 0.5 * self.length:
            self.x = 0 + 2 + 0.5 * self.length

        if self.x >= height - 1.2 * self.length:
            self.x = height - 2 - 1.2 * self.length

        if self.y < 0 + 0.5 * self.width:
            self.y = 0 + 2 + 0.5 * self.width

        if self.y >= width - 1.2 * self.width:
            self.y = width - 2 - 1.2 * self.width

        self.rotated = pygame.transform.rotozoom(self.img,
                                                 math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        return

    def get_sensor(self, obstacles, map):

        if self.sensor[0] == 0:

            self.sensor = [0, 0, 0, 0, 0, 0]

            for obstacle in obstacles:
                dist_to_wall = calc_distance((self.x, self.y),
                                             (obstacle.x, obstacle.y))

                if dist_to_wall <= 70:
                    angle_w_wall = calc_angle((self.x, self.y),
                                              (obstacle.x, obstacle.y))

                    if 0 <= angle_w_wall < (math.pi * 2 * 1 / 5):
                        # pygame.draw.line(map, (255, 0, 0), (self.x, self.y),
                        #                  (obstacle.x, obstacle.y))
                        self.sensor[1] = 1  #* 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 1 / 5) <= angle_w_wall < (math.pi * 2 *
                                                                  2 / 5):
                        # pygame.draw.line(map, (255, 0, 0), (self.x, self.y),
                        #                  (obstacle.x, obstacle.y))
                        self.sensor[2] = 1 #* 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 2 / 5) <= angle_w_wall < (math.pi * 2 *
                                                                  3 / 5):
                        # pygame.draw.line(map, (255, 0, 0), (self.x, self.y),
                        #                  (obstacle.x, obstacle.y))
                        self.sensor[3] = 1 #* 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 3 / 5) <= angle_w_wall < (math.pi * 2 *
                                                                  4 / 5):
                        # pygame.draw.line(map, (255, 0, 0), (self.x, self.y),
                        #                  (obstacle.x, obstacle.y))
                        self.sensor[4] = 1 #* 1 / max(1, dist_to_wall)

                    else:
                        # pygame.draw.line(map, (255, 0, 0), (self.x, self.y),
                        #                  (obstacle.x, obstacle.y))
                        self.sensor[5] = 1 #* 1 / max(1, dist_to_wall)

    def get_reward(self):

        self.avg_dist = math.sqrt((self.init_pos[0] - self.x)**2 +
                                  (self.init_pos[1] - self.y)**2)
       
        # score = (self.token * 8 + self.dist_travelled * 2 - self.collision * 1) / (
        #     self.token  + self.dist_travelled + self.collision)

        score = (self.dist_travelled / self.m2p)*2 + self.collision * - 2 + self.token * 5
        fitness = score

        return float(fitness)