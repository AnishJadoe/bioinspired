import numpy as np
import pygame
import math
from map_generator.txt_to_map import WorldMap
from functions import calc_distance
from functions import calc_angle
from constants import *

def find_closest_cell(agent_location, movable_cells):
    # Extract the coordinates of the agent's location
    agent_x, agent_y = agent_location
    
    # Initialize variables to keep track of the closest cell index and its distance
    closest_index = None
    min_distance = float('inf')
    
    # Iterate through movable cells to find the closest one
    for index, cell in enumerate(movable_cells):
        # Extract the coordinates of the movable cell
        cell_x, cell_y, _, _ = cell
        
        # Calculate the Euclidean distance between the agent and the cell
        distance = math.sqrt((agent_x - cell_x)**2 + (agent_y - cell_y)**2)
        
        # Update the closest cell index and distance if necessary
        if distance < min_distance:
            min_distance = distance
            closest_index = index
    
    return closest_index

class Robot:
    def __init__(self, startpos, width, chromosome):

        self.m2p = 3779.52  # meters 2 pixels
        self.w = width * 20
        
        self.init_pos = startpos
        self.x = startpos[0]
        self.y = startpos[1]
        self.current_tile = 0
        self.theta = 0
        self.vl = 1
        self.vr = 1
        self.maxspeed = 255
        self.minspeed = -255
        self.all_states = np.array(np.meshgrid([0, 1],[0,1],[0,1],[0,1],[0,1])).T.reshape(-1, 5)
        self.wall_collided = -1
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

        self.width = width 
        self.length = width 
        self.collision = 0
        self.token = 0

        self.base_img = pygame.image.load("Images/robot.png")
        self.base_img = pygame.transform.scale(self.base_img, (self.width, self.length))
        self.base_img = pygame.transform.rotate(self.base_img, math.degrees(self.theta))
        self.sensor_on = pygame.image.load("Images/distance_sensor_on.png")
        self.sensor_on = pygame.transform.scale(self.sensor_on,(50,80))

        self.trans_img = self.base_img
        self.hitbox = self.trans_img.get_rect(center=(self.x, self.y))

        self.chromosome = chromosome

    def draw_sensor(self,world):
        for i,sensor in enumerate(self.sensor):
            if sensor:
                sensor_image = self.sensor_on      
                sensor_image = pygame.transform.rotate(sensor_image, math.degrees(self.theta) + i* (360//5))
                sensor_rect_obj = sensor_image.get_rect(center=(self.x,self.y))
                world.blit(sensor_image, sensor_rect_obj)

    def debug_theta(self,world):
        font = pygame.font.SysFont(None, 24)
        img = font.render(f'theta: {round(math.degrees(self.theta),1)}', True, (0,0,0))
        world.blit(img,(self.x,self.y))

    def draw_robot(self,world):
        self.trans_img = pygame.transform.rotozoom(self.base_img,
                                                 math.degrees(self.theta), 1)
        self.hitbox = self.trans_img.get_rect(center=(self.x, self.y))
        world.blit(self.trans_img, self.hitbox)

    def draw(self, world):
        '''
        Draws on the world
        '''
        self.draw_robot(world)
        self.draw_sensor(world)
        self.debug_theta(world)

        
    def get_collision(self, world_map: WorldMap):
        """Function to calculate whetere collisions between the walls and the agent have occured
        """
        NO_OBSTACLES = -1
        obstacles = self.hitbox.collidelist(world_map.walls)
        if obstacles != NO_OBSTACLES: # -1 equals no collision:
            self.collision += 1
            self.sensor[0] = 1
            return world_map.walls[obstacles]
        else:
            return 0
        

    def get_token(self, tokens):
        '''
        Function to check if we have collided with a token, when this happens the
        robot obtains a token
        '''
        for token in tokens:
            if self.hitbox.colliderect(token):
                tokens.pop(token)
                self.token += 1

    def move(self, wall_collided, dt, event=None, auto=False):

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


        if auto:
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

        if self.theta >= 2*math.pi:
            self.theta = 0

        if self.sensor[0]:
            # Approach wall from Top Left
            if self.x > wall_collided.x \
                    and self.y > wall_collided.y:  
                self.x += 1
                self.y += 1
            # Approach wall from Top Left
            elif self.x <= wall_collided.x \
                    and self.y > wall_collided.y:  
                self.x -= 1
                self.y += 1
            # Approach wall from Bottom Left
            elif self.x > wall_collided.x \
                    and self.y <= wall_collided.y:  
                self.x += 1
                self.y -= 1
            # Approach wall from Bottom Right
            elif self.x <= wall_collided.x \
                    and self.y <= wall_collided.y:  
                self.x -= 1
                self.y -= 1
            else:  # Edge case, assume flip whole car
                self.theta += math.pi * 0.5 
        else:
            self.x += (((self.vl + self.vr) / 2) * math.cos(self.theta) *
                        dt)
            self.y -= (((self.vl + self.vr) / 2) * math.sin(self.theta) *
                        dt)
        
        # Detects if we're within map borders
        if self.x < 0 + 0.5 * self.length:
            self.x = 0 + 2 + 0.5 * self.length

        if self.x >= MAP_SIZE[0] - 1.2 * self.length:
            self.x = MAP_SIZE[0] - 2 - 1.2 * self.length

        if self.y < 0 + 0.5 * self.width:
            self.y = 0 + 2 + 0.5 * self.width

        if self.y >= MAP_SIZE[1] - 1.2 * self.width:
            self.y = MAP_SIZE[1] - 2 - 1.2 * self.width

        return

    def find_obstacles(self, FOV):
        
        self.sensor = [0, 0, 0, 0, 0, 0]

        CASE_3 = self.theta >= -2 * (2*math.pi / 5) and self.theta < -1 * (2*math.pi / 5)
        CASE_2 = self.theta >= -1 * (2*math.pi / 5) and self.theta < 0 * (2*math.pi / 5)
        CASE_1 = self.theta >= 0 * (2*math.pi / 5) and self.theta < 1 * (2*math.pi / 5)
        CASE_4 = self.theta >= 1 * (2*math.pi / 5) and self.theta < 2 * (2*math.pi / 5)
        CASE_5 = self.theta >= 2 * (2*math.pi / 5) and self.theta <= 3 * (2*math.pi / 5)

        if FOV[0:3, 0:3].sum() >= 1:  # Top left
        
            if CASE_1:
                self.sensor[5] = 1
            if CASE_2:
                self.sensor[1] = 1
            if CASE_3:
                self.sensor[2] = 1
            if CASE_4:
                self.sensor[3] = 1
            if CASE_5:
                self.sensor[4] = 1
        
        if FOV[0:3, 2].sum() >= 1:  # Center
            if CASE_1:
                self.sensor[1] = 1
            if CASE_2:
                self.sensor[2] = 1
            if CASE_3:
                self.sensor[3] = 1
            if CASE_4:
                self.sensor[4] = 1
            if CASE_5:
                self.sensor[5] = 1

        if FOV[0:3, 3:-1].sum() >= 1:  # Top Right
            if CASE_1:
                self.sensor[2] = 1
            if CASE_2:
                self.sensor[3] = 1
            if CASE_3:
                self.sensor[4] = 1
            if CASE_4:
                self.sensor[5] = 1
            if CASE_5:
                self.sensor[1] = 1

        if FOV[3:-1, 0:3].sum() >= 1:  # Bot right
            if CASE_1:
                self.sensor[3] = 1
            if CASE_2:
                self.sensor[4] = 1
            if CASE_3:
                self.sensor[5] = 1
            if CASE_4:
                self.sensor[1] = 1
            if CASE_5:
                self.sensor[2] = 1

        if FOV[3:-1, 3:-1].sum() >= 1:  # Bot left
            if CASE_1:
                self.sensor[4] = 1
            if CASE_2:
                self.sensor[5] = 1
            if CASE_3:
                self.sensor[1] = 1
            if CASE_4:
                self.sensor[2] = 1
            if CASE_5:
                self.sensor[3] = 1

        return
    
    def get_reward(self):

        self.avg_dist = math.sqrt((self.init_pos[0] - self.x)**2 +
                                  (self.init_pos[1] - self.y)**2)
        
        return (self.dist_travelled / self.m2p)*10 + \
            self.token * 100 - self.collision# + len(self.visited_cells) * 15
          
    def get_FOV(self, map):
        FOV_size = 2
        current_pos_x = int(self.x/20)
        current_pos_y = int(self.y/20)
        FOV = map.binary_map[current_pos_x-FOV_size : current_pos_x+FOV_size+1, current_pos_y-FOV_size : current_pos_y+FOV_size+1]
        if FOV.shape != (5,5):
            FOV = np.ones((7,7))
        return FOV

        