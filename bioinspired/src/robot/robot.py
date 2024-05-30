import numpy as np
import pygame
import math

from ..utility.functions import calc_angle, calc_distance
from ..world_map.txt_to_map import WorldMap
from ..utility.constants import *

WHITE = (255,255,255,128)
DARK_GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)
ORANGE = (255, 165, 0)
SENSOR_COLORS = [RED,DARK_GREEN,PURPLE,ORANGE,BLUE]

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
    def __init__(self,  startpos, width, chromosome, token_locations, special_flag,robot_id=1):
        self.id = robot_id + 1
        self.chromosome = chromosome
        self.m2p = 3779.52  # meters 2 pixels
        self.w = width * 20
        self.tokens_locations = token_locations

        self.init_pos = startpos
        self.x = startpos[0]
        self.y = startpos[1]
        self.current_tile = 0
        self.theta = 0
        self.vl = 0
        self.vr = 0
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
        self.sensor_spacing = [0, 30, 60, -30, -60]


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
        self.special = special_flag

        # Build robot images
        if self.special:
            self.base_img = pygame.image.load(r"bioinspired\src\robot\images\robot_special.png")
            self.base_img = pygame.transform.scale(self.base_img, (self.width, self.length))
            self.base_img = pygame.transform.rotate(self.base_img, 0)
        else:   
            self.base_img = pygame.image.load(r"bioinspired\src\robot\images\robot_normal.png")
            self.base_img = pygame.transform.scale(self.base_img, (self.width, self.length))
            self.base_img = pygame.transform.rotate(self.base_img, 0)
        
        self.trans_img = self.base_img
        self.hitbox = self.trans_img.get_rect(center=(self.x, self.y))
        
        # Initialize sensor image 
        self.sensor_on_img = pygame.image.load(r"bioinspired\src\robot\images\distance_sensor_on.png")
        self.sensor_off_img = pygame.image.load(r"bioinspired\src\robot\images\distance_sensor_off.png")
        self.sensor_on_img = pygame.transform.scale(self.sensor_on_img,(20,100))
        self.sensor_off_img = pygame.transform.scale(self.sensor_off_img,(20,100))

        self.sensor_imgs = []
        self.sensor_hitboxes = []
        for i in range(5):
            self.sensor_imgs.append(pygame.transform.rotate(self.sensor_off_img, self.sensor_spacing[i]))
            self.sensor_hitboxes.append(self.sensor_imgs[i].get_rect(center=(self.x, self.y)))

    def draw_sensor(self,world):
        for i,sensor in enumerate(self.sensor[1:2]):
            rotation = self.sensor_spacing[i] #+ math.degrees(self.theta)
            if sensor:  
                self.sensor_imgs[i] = pygame.transform.rotozoom(self.sensor_on_img,rotation , 1)
            else:
                self.sensor_imgs[i] = pygame.transform.rotozoom(self.sensor_off_img,rotation, 1)

            self.sensor_hitboxes[i] = self.sensor_imgs[i].get_rect(center=(500, 250))
            pygame.draw.rect(world,RED,self.sensor_hitboxes[i], width=1)

            world.blit(self.sensor_imgs[i], self.sensor_hitboxes[i])
  
    def debug_theta(self,world):
        font = pygame.font.SysFont(None, 24)
        img = font.render(f'theta: {round(math.degrees(self.theta),1)}', True, (0,0,0))
        world.blit(img,(self.x,self.y))

    def debug_token(self,world):
        font = pygame.font.SysFont(None, 24)
        img = font.render(f'tokens: {self.token}', True, (0,0,0))
        world.blit(img,(self.x,self.y))

    def draw_robot(self,world):

        self.trans_img = pygame.transform.rotozoom(self.base_img,
                                                 math.degrees(self.theta), 1)
        self.hitbox = self.trans_img.get_rect(center=(self.x, self.y))
        world.blit(self.trans_img, self.hitbox)

    def draw_robot_bb(self,world):
        pygame.draw.rect(world,RED,self.hitbox, width=1)
        
    def draw(self, world):
        '''
        Draws on the world
        '''
        self.draw_robot(world)
        # self.draw_robot_bb(world)
        #self.debug_token(world)
        #self.draw_sensor(world)
        # self.debug_theta(world)

    # def update_sensors(self, nearby_obstacles, world_map):
    #         self.sensor = [0, 0, 0, 0, 0, 0]
    #         sensor_range = 45

    #         for obstacle in nearby_obstacles:
    #             dist_to_wall = calc_distance((self.x, self.y), (obstacle.x, obstacle.y))
    #             if dist_to_wall <= sensor_range:
    #                 angle_w_wall = calc_angle((self.x, self.y), (obstacle.x, obstacle.y))
    #                 relative_angle = (angle_w_wall - self.theta) % (2 * math.pi)

    #                 for i, sensor_angle in enumerate(self.sensor_spacing):
    #                     sensor_start = (sensor_angle - 22.5) * (math.pi / 180)
    #                     sensor_end = (sensor_angle + 22.5) * (math.pi / 180)
    #                     if sensor_start <= relative_angle < sensor_end:
    #                         pygame.draw.line(world_map.surf, SENSOR_COLORS[i], (self.x, self.y), (self.x+math.cos(sensor_angle-self.theta)*sensor_range,
    #                         self.y+math.sin(sensor_angle-self.theta)*sensor_range),width=5)
    #                         self.sensor[i + 1] = 1
    #                         break

    #         return
    
        
    def update_sensors(self, nearby_obstacles, world_map:WorldMap):
        self.sensor = [0, 0, 0, 0, 0, 0]
        sensor_range = 30
        start_point = (self.x,self.y)
        sensors = [pygame.draw.line(world_map.surf,BLUE,start_point, 
                            (self.x+math.cos(spacing-self.theta)*sensor_range,
                            self.y+math.sin(spacing-self.theta)*sensor_range), width=1) for spacing in self.sensor_spacing]
        
        for i in range(len(self.sensor)-1):
            walls_near = sensors[i].collidelist(nearby_obstacles)
            if walls_near > 0:
                self.sensor[i+1] = 1
        return
     

    def get_collision(self, nearby_obstacles):
        """Function to calculate whetere collisions between the walls and the agent have occured
        """
        NO_COLLISIONS = -1

        collided_walls = self.hitbox.collidelist(nearby_obstacles)
        if collided_walls != NO_COLLISIONS: # -1 equals no collision:
            self.collision += 1
            self.sensor[0] = 1
            return nearby_obstacles[collided_walls]
        else:
            return 0
        

    def get_tokens(self):
        '''
        Function to check if we have collided with a token, when this happens the
        robot obtains a token
        '''
        tokens_collected = self.hitbox.collidelist(self.tokens_locations) 
        if tokens_collected == -1:
            return
        self.token += 1
        self.tokens_locations.pop(tokens_collected)
        return

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
            self.vl = actions[0] 
            self.vr = actions[1] 
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
    
    def get_reward(self):
        return ((self.dist_travelled/self.m2p) * (1+self.token) - self.collision/50)
    
    def find_nearby_obstacles(self,world_map:WorldMap):
        nearby_obstacles = []
        cell_x = int(self.x // world_map.tile_size)
        cell_y = int(self.y // world_map.tile_size)
        for i in range(max(0, cell_x - 3), min(world_map.map_width, cell_x + 4)):
            for j in range(max(0, cell_y - 3), min(world_map.map_height, cell_y + 4)):
                if world_map.binary_map[i][j] == 1:
                    nearby_obstacles.extend(world_map.spatial_grid[i][j])
        return nearby_obstacles


        