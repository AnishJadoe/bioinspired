import cProfile
import numpy as np
import pygame
import math
import pygame.locals

from ..utility.functions import calc_angle, calc_distance
from ..world_map.txt_to_map import WorldMap
from ..utility.constants import *
from ..robot.neural_net import NeuralNet

def bin_angle(angle, bin_width=0.1):
        """Bins the angle to the nearest bin width."""
        return round(angle / bin_width) * bin_width


def find_closest_cell(agent_location, cells):
    # Extract the coordinates of the agent's location
    agent_x, agent_y = agent_location
    
    # Initialize variables to keep track of the closest cell index and its distance
    closest_index = None
    min_distance = float('inf')
    
    # Iterate through movable cells to find the closest one
    for index, cell in enumerate(cells):
        # Extract the coordinates of the movable cell
        cell_x , cell_y, _, _ = cell
        
        # Calculate the Euclidean distance between the agent and the cell
        distance = math.sqrt((agent_x - cell_x // CELL_SIZE)**2 + (agent_y - cell_y // CELL_SIZE)**2)
        
        # Update the closest cell index and distance if necessary
        if distance < min_distance:
            min_distance = distance
            closest_index = index
    
    return closest_index

class Robot:
    def __init__(self,  startpos, width, chromosome, token_locations, special_flag,robot_id=1):
        self.id = robot_id + 1
        self.chromosome = chromosome
        self.controller = NeuralNet(n_inputs=N_INPUTS,n_outputs=N_OUTPUTS,n_hidden=N_HIDDEN,chromosome=self.chromosome)
        self.m2p = 3779.52  # meters 2 pixels
        self.w = width * 20
        self.tokens_locations = token_locations

        self.init_pos = startpos
        self.x = startpos[0]
        self.y = startpos[1]
        self.delta_x = 0 # difference between current x and desired x
        self.delta_y = 0 # difference between current y and desired y
        self.theta = 0
        
        self.vl = 0
        self.vr = 0
        self.maxspeed = 255
        self.minspeed = -255
        self.all_states = np.array(np.meshgrid([0, 1],[0,1],[0,1],[0,1],[0,1], [0,1])).T.reshape(-1, 6)
        self.wall_collided = -1
        self.ls_tot_speed = list()
        self.visited_cells = set()
        self.ls_theta = []
        self.ls_x = []
        self.ls_y = []
        self.omega = 0

        self.dist_travelled = 0
        self.avg_dist = 0

        self.sensor_spacing = [math.radians(-150),math.radians(-60) , 
                               math.radians(0), math.radians(60), math.radians(150)]
        self.sensor = [
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # 1 tactile sensor (sensor[0]) and 5 ultrasound
        self.sensor_range = 45
        self.sensor_sweep = math.radians(25) # degrees
        # Precompute sensor angle ranges
        self.sensor_ranges = [(sensor_angle - self.sensor_sweep, sensor_angle + self.sensor_sweep) 
                        for sensor_angle in self.sensor_spacing]
        
        self.state = np.array([self.delta_x,self.delta_y,self.theta, *self.sensor]).reshape(-1,1)
        
        self.width = width 
        self.length = width 
        self.collision = 0
        self.token = 0
        self.special = special_flag
        self.looking_for_token = False
        self.closest_token = self.tokens_locations[0]

        # Build robot images
        if self.special:
            self.base_img = pygame.image.load(r"bioinspired/src/robot/images/robot_special.png")
            self.base_img = pygame.transform.scale(self.base_img, (self.width, self.length))
            self.base_img = pygame.transform.rotate(self.base_img, 0)
        else:   
            self.base_img = pygame.image.load(r"bioinspired/src/robot/images/robot_normal.png")
            self.base_img = pygame.transform.scale(self.base_img, (self.width, self.length))
            self.base_img = pygame.transform.rotate(self.base_img, 0)
        
        self.trans_img = self.base_img
        self.hitbox = self.trans_img.get_rect(center=(self.x, self.y))
        
        # Initialize sensor image 
        self.sensor_on_img = pygame.image.load(r"bioinspired/src/robot/images/distance_sensor_on.png")
        self.sensor_off_img = pygame.image.load(r"bioinspired/src/robot/images/distance_sensor_off.png")
        self.sensor_on_img = pygame.transform.scale(self.sensor_on_img,(20,100))
        self.sensor_off_img = pygame.transform.scale(self.sensor_off_img,(20,100))

    def _attitude(self):
        return(self.x,self.y,math.degrees(self.theta))
    
    def draw_sensor_orientation(self,world):
        for i,sensor in enumerate(self.sensor[1:]):
            sensor_angle = self.sensor_spacing[i]
            pygame.draw.line(world, SENSOR_COLORS[i], (self.x, self.y), 
            (self.x+math.cos(self.theta+math.radians(sensor_angle))*self.sensor_range,
            self.y+math.sin(self.theta+math.radians(sensor_angle))*self.sensor_range),width=2)

            pygame.draw.line(world, SENSOR_COLORS[i], (self.x, self.y), 
            (self.x+math.cos(self.theta+math.radians(sensor_angle + self.sensor_sweep))*self.sensor_range,
            self.y+math.sin(self.theta+math.radians(sensor_angle + self.sensor_sweep))*self.sensor_range),width=2)

            pygame.draw.line(world, SENSOR_COLORS[i], (self.x, self.y), 
            (self.x+math.cos(self.theta+math.radians(sensor_angle - self.sensor_sweep))*self.sensor_range,
            self.y+math.sin(self.theta+math.radians(sensor_angle - self.sensor_sweep))*self.sensor_range),width=2)

    def draw_sensor_activation(self,world):
        for i,sensor in enumerate(self.sensor[1:]):
            if sensor:
                sensor_angle = self.sensor_spacing[i]
                pygame.draw.line(world, SENSOR_COLORS[i], (self.x, self.y), 
                (self.x+math.cos(self.theta+math.radians(sensor_angle))*self.sensor_range,
                self.y+math.sin(self.theta+math.radians(sensor_angle))*self.sensor_range),width=6)

                pygame.draw.line(world, SENSOR_COLORS[i], (self.x, self.y), 
                (self.x+math.cos(self.theta+math.radians(sensor_angle + self.sensor_sweep))*self.sensor_range,
                self.y+math.sin(self.theta+math.radians(sensor_angle + self.sensor_sweep))*self.sensor_range),width=6)

                pygame.draw.line(world, SENSOR_COLORS[i], (self.x, self.y), 
                (self.x+math.cos(self.theta+math.radians(sensor_angle - self.sensor_sweep))*self.sensor_range,
                self.y+math.sin(self.theta+math.radians(sensor_angle - self.sensor_sweep))*self.sensor_range),width=6)

    def debug_theta(self,world):
        font = pygame.font.SysFont(None, 24)
        img = font.render(f'theta: {round(math.degrees(self.theta),1)}', True, (0,0,0))
        world.blit(img,(self.x,self.y))
        # img = font.render(f'vr: {round(self.vr,2)}' , True, (0,0,0))
        # world.blit(img,(400,300))
        # img = font.render(f'vl: {round(self.vl,2)}', True, (0,0,0))
        # world.blit(img,(400,200))
        pygame.draw.line(world, RED, (self.x, self.y), 
        (self.x+math.cos(self.theta)*self.sensor_range,
        self.y+math.sin(self.theta)*self.sensor_range),width=2)


    def debug_token(self,world):
        font = pygame.font.SysFont(None, 24)
        img = font.render(f'tokens: {self.token}', True, (0,0,0))
        world.blit(img,(self.x,self.y))

    def draw_token(self,world):
        font = pygame.font.SysFont(None, 24)
        img = font.render(f'Tokens collected: {self.token}', True, (0,0,0))
        world.blit(img,(500,100))    

    def draw_visited_cells(self,world):
        font = pygame.font.SysFont(None, 24)
        img = font.render(f'Visited Cells: {len(self.visited_cells)}', True, (0,0,0))
        world.blit(img,(self.x,self.y))  

    def draw_robot(self,world, debug=False):
        self.trans_img = pygame.transform.rotozoom(self.base_img,
                                                 math.degrees(-self.theta), 1)
        self.hitbox = self.trans_img.get_rect(center=(self.x, self.y))
        world.blit(self.trans_img, self.hitbox)
        # img = font.render(f'ID: {self.id}', True, (0,0,0))
        # world.blit(img,(self.x,self.y-20))    

    def draw_robot_bb(self,world):
        pygame.draw.rect(world,RED,self.hitbox, width=1)
        
    def draw(self, world):
        '''
        Draws on the world
        '''
        self.draw_robot(world)
        # self.draw_visited_cells(world)
        # self.draw_robot_bb(world)
        #self.debug_token(world)
        # self.draw_sensor_orientation(world)
        # self.draw_sensor_activation(world)
        # self.debug_theta(world)
        # self.draw_token(world)

    def update_sensors(self, nearby_obstacles, world_map):
        # Reset sensors
        self.sensor = [0, 0, 0, 0, 0, 0]
        tile_size = world_map.tile_size
        # Precompute cos and sin of theta
        rounded_theta = bin_angle(self.theta,1)
        cos_theta = math.cos(rounded_theta)
        sin_theta = math.sin(rounded_theta)
        
        for obstacle in nearby_obstacles:
            # Calculate distance to obstacle
            v2_x = obstacle.x // tile_size - self.x //tile_size
            v2_y = obstacle.y // tile_size - self.y // tile_size
            dist_to_wall = np.hypot(v2_x, v2_y)
            if dist_to_wall <= self.sensor_range:
                # Calculate the angle between robot direction and obstacle
                relative_angle = calc_angle((cos_theta, sin_theta), (v2_x, v2_y))
                # Check which sensor should be activated
                for i, (sensor_start, sensor_end) in enumerate(self.sensor_ranges):
                    if sensor_start <= relative_angle < sensor_end:
                        self.sensor[i + 1] = 1
                        break
        
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
        
    def get_action(self):
        return self.controller.forward_pass(self.state)
    
        
    def get_reference_position(self):
            
     
        closest_cell_index = find_closest_cell((self.x // CELL_SIZE, self.y // CELL_SIZE), self.tokens_locations)
        self.closest_token = self.tokens_locations[closest_cell_index]
        self.looking_for_token = True
        
        self.delta_x = round(self.closest_token.x - self.x)
        self.delta_y = round(self.closest_token.y - self.y)
    
        # if self.closest_token not in self.tokens_locations:
        #     self.looking_for_token = False
        
        
    def update_state(self):
        self.get_reference_position()
        self.state = np.array([self.delta_x,self.delta_y,bin_angle(self.theta,1), *self.sensor]).reshape(-1,1)
        return
        
    def get_tokens(self):
        '''
        Function to check if we have collided with a token, when this happens the
        robot obtains a token
        '''
        token_collected = self.hitbox.collidelist(self.tokens_locations) 
        if token_collected == -1:
            return
        self.token += 1
        self.tokens_locations.pop(token_collected)
            
        return

    def move(self, wall_collided, dt, event=None, auto=False):
        self.ls_x.append(self.x)
        self.ls_y.append(self.y)
        self.ls_theta.append(self.theta)

        if len(self.ls_x) >= 2:
            delta_x = self.ls_x[-2] - self.ls_x[-1]
            delta_y = self.ls_y[-2] - self.ls_y[-1]
            delta_r = np.hypot(delta_x, delta_y)  # Efficient calculation of sqrt(delta_x^2 + delta_y^2)
            self.dist_travelled += delta_r

        if auto:
            # index = np.where((self.all_states == self.sensor).all(axis=1))[0][0]
            # actions = self.chromosome[index]
            actions = self.get_action()
            self.vl = actions[0,:][0] 
            self.vr = actions[1,:][0] 
        elif event:
            if event.type in {pygame.KEYDOWN, pygame.KEYUP}:
                if event.key == pygame.locals.K_a:
                    self.vl += 0.01 * self.m2p
                elif event.key == pygame.locals.K_s:
                    self.vl -= 0.01 * self.m2p
                elif event.key == pygame.locals.K_k:
                    self.vr += 0.01 * self.m2p
                elif event.key == pygame.locals.K_j:
                    self.vr -= 0.01 * self.m2p

        self.omega = (self.vl - self.vr) / self.w
        self.theta += self.omega * dt

        if self.theta >= math.pi:
            self.theta -= 2 * math.pi

        wall_elasticity = 0.5
        if self.sensor[0]:
            if self.x > wall_collided.x and self.y > wall_collided.y:
                self.x += wall_elasticity + 1
                self.y += wall_elasticity
            elif self.x <= wall_collided.x and self.y >= wall_collided.y:
                self.x -= wall_elasticity
                self.y += wall_elasticity + 1
            elif self.x >= wall_collided.x and self.y <= wall_collided.y:
                self.x += wall_elasticity
                self.y -= wall_elasticity + 1
            elif self.x <= wall_collided.x and self.y <= wall_collided.y:
                self.x -= wall_elasticity + 1
                self.y -= wall_elasticity
            else:
                self.theta += math.pi * 0.5
        else:
            cos_theta = math.cos(-self.theta)
            sin_theta = math.sin(-self.theta)
            avg_velocity = (self.vl + self.vr) / 2
            self.x += avg_velocity * cos_theta * dt
            self.y -= avg_velocity * sin_theta * dt

        half_length = 0.5 * self.length
        half_width = 0.5 * self.width
        if self.x < half_length:
            self.x = half_length + 2
        elif self.x >= MAP_SIZE[0] - 1.2 * self.length:
            self.x = MAP_SIZE[0] - 1.2 * self.length - 2

        if self.y < half_width:
            self.y = half_width + 2
        elif self.y >= MAP_SIZE[1] - 1.2 * self.width:
            self.y = MAP_SIZE[1] - 1.2 * self.width - 2

        return
    
    def get_reward(self):
        return round(5*(self.dist_travelled/self.m2p) + 10*self.token + 0.1*len(self.visited_cells) - self.collision*0.3,2)
    
    # def get_reward(self):
    #     return (self.dist_travelled/self.m2p) * (1+self.token) - (self.collision/len(self.visited_cells))
    
    def find_position(self,world_map:WorldMap):
        nearby_obstacles = []
        cell_x = int(self.x // world_map.tile_size)
        cell_y = int(self.y // world_map.tile_size)
        min_range_x = max(0, cell_x - 3)
        max_range_x = min(world_map.map_width, cell_x + 4)
        min_range_y = max(0, cell_y - 3)
        max_range_y = min(world_map.map_height, cell_y + 4)

        self.visited_cells.add((cell_x,cell_y))
        for i in range(min_range_x, max_range_x):
            for j in range(min_range_y, max_range_y):
                if world_map.binary_map[i][j] == 1:
                    nearby_obstacles.extend(world_map.spatial_grid[i][j])
        return nearby_obstacles
    


        