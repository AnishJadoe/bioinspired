import math

import pygame
import numpy as np
from src.utility.functions import bin_angle, bound, calc_angle, calc_distance, find_closest_cell, find_nearby_obstacles
from src.world_map.world_map import Target, Tile, WorldMap
from ..utility.constants import *
from ..controllers.controllers import ManualController, NeuroController

ROBOT_WIDTH = 25
MAX_SPEED = 128

token_map_cache = {}
nearby_items_cache = {}
wall_cache = {}
token_cache = {}

def find_tile_id(tiles,tile):
    id_list = [target.id for target in tiles]
    if tile.id in id_list:
        return id_list.index(tile.id)
    else:
        return -1

def build_token_map(token_locations):
    # Convert token locations to a hashable representation (tuple of coordinates)
    token_key = tuple((token.x, token.y) for token in token_locations)
    
    if token_key in token_map_cache:
        return token_map_cache[token_key]
    
    token_map = np.zeros((MAP_DIMS[0], MAP_DIMS[1]))
    for token in token_locations:
        token_map[token.x // CELL_SIZE][token.y // CELL_SIZE] = 1
    
    token_map_cache[token_key] = token_map
    return token_map

class BaseRobot():
    def __init__(self, startpos, targets,end_target, robot_id, special=False):
        self.id = robot_id
        self.startpos = startpos.hitbox
        self.x = startpos.hitbox.x
        self.y = startpos.hitbox.y
        self.theta = 0
        self.omega = 0
        self.vl = 0
        self.vr = 0
        self.avg_velocity = (self.vr+self.vl) / 2
        self.mass = 1e-5
        self.sensor = [0,0,0,0,0,0]
        self.size = (ROBOT_WIDTH,ROBOT_WIDTH)
        self.targets = targets
        self.collided = False
        self.mission_complete = False
        self.targets_collected = 0
        self.moment_of_inertia = (10*self.size[0])
        self.collided_w_wall = 0
        self.log_attitude = []
        self.log_rates = []
        
        self.current_target = None
        self.current_target_id = -1
        
        if end_target:
            self.end_target = end_target
            self.targets.append(end_target)

        self._init_images(special)
        self.trans_img = self.base_img
        self.img = self.base_img
        self.hitbox = self.img.get_rect(center=(self.x, self.y))
    
    def _init_images(self, special):
        # Image loading logic moved to a separate method
        if special and self.id != 1:
            self.base_img = pygame.image.load(r"bioinspired/src/robots/images/robot_special.png")
        elif self.id == 1:
            self.base_img = pygame.image.load(r"bioinspired/src/robots/images/robot_one.png")
        else:
            self.base_img = pygame.image.load(r"bioinspired/src/robots/images/robot_normal.png")
        
        self.base_img = pygame.transform.scale(self.base_img, (self.size[0], self.size[1]))
        self.base_img = pygame.transform.rotate(self.base_img, 0)
        self.tank_empty_img = pygame.image.load(r"bioinspired/src/robots/images/robot_dead.png")
        self.tank_empty_img = pygame.transform.scale(self.tank_empty_img, (self.size[0], self.size[1]))
        self.tank_empty_img = pygame.transform.rotate(self.tank_empty_img, 0)
    
    def _get_init_state(self):
        raise NotImplementedError
    
    def _get_collision(self, nearby_obstacles):
        """Function to calculate whetere collisions between the walls and the agent have occured"""
        NO_COLLISIONS = -1

        collided_walls = self.hitbox.collidelist([obstacle.hitbox for obstacle in nearby_obstacles])
        if collided_walls != NO_COLLISIONS: # -1 equals no collision:
            self.collided = True
            return nearby_obstacles[collided_walls]
        else:
            self.collided = False
            return 
    
    def _update_state(self, dt: float, wall_collided:pygame.rect.Rect):
        raise NotImplementedError

    
    def _handle_collision(self, wall_collided: pygame.rect.Rect):
        # TODO VECTORIZE THIS, should be easy. Calculate unit vector of collision, reverse it and then add an angle to it 
        # such that the robot does not keep bouncing off the same walls in a corner
        wall_elasticity = 1
        x = self.hitbox.x
        y = self.hitbox.y
        #  Collided with wall from the top left
        if x >= wall_collided.x and y  >= wall_collided.y:
            self.x += wall_elasticity + 0.2
            self.y += wall_elasticity
        # Collided with wall from the top right
        elif x <= wall_collided.x and y  >= wall_collided.y:
            self.x -= wall_elasticity
            self.y += wall_elasticity + 0.2
        # Collided with wall from the bottom left
        elif x >= wall_collided.x and y  <= wall_collided.y:
            self.x += wall_elasticity
            self.y -= wall_elasticity + 0.2
        # Collided with wall from the top right
        elif x  <= wall_collided.x and y <= wall_collided.y:
            self.x -= wall_elasticity + 0.2
            self.y -= wall_elasticity
            
        elif x  <= wall_collided.x and y  == wall_collided.y:
            self.x -= wall_elasticity 
        elif x   >= wall_collided.x and y  == wall_collided.y:
            self.x += wall_elasticity 
        elif x   == wall_collided.x and y  <= wall_collided.y:
            self.y -= wall_elasticity 
        elif x  == wall_collided.x and y  >= wall_collided.y:
            self.y += wall_elasticity 
        # else:
        #     self.theta += math.pi * 0.5
            
        return
    
    def _handle_map_boundaries(self):
        half_length = 0.5 * self.size[0]
        half_width = 0.5 * self.size[1]
        if self.x < half_length:
            self.x = half_length + 2
        elif self.x >= MAP_SIZE[0] - 1.2 * self.size[1]:
            self.x = MAP_SIZE[0] - 1.2 * self.size[1] - 2

        if self.y < half_width:
            self.y = half_width + 2
        elif self.y >= MAP_SIZE[1] - 1.2 * self.size[0]:
            self.y = MAP_SIZE[1] - 1.2 * self.size[0] - 2
    
    def update_tank(self):
        raise NotImplementedError
    
    def handler(self, dt, nearby_obstacles):
        self._handle_map_boundaries()
        self._update_state(dt, nearby_obstacles)
    
    def _get_task(self):
        raise NotImplementedError
    
    def _check_task(self):
        '''
        Function to check if we have collided with a token, when this happens the
        robot obtains a token
        '''
        target_reached = self.hitbox.collidelist(self.targets) 
        if target_reached == -1:
            return 
        if self.current_target == self.targets[target_reached]:
            self.targets.pop(target_reached)
            if len(self.targets) == 0:
                self.mission_complete = True
                return 
            self.current_target = self._get_new_task()
        return 
    
    def _get_new_task(self):
        if not self.mission_complete:
            closest_cell_index = find_closest_cell((self.x // CELL_SIZE, self.y // CELL_SIZE), [target.rect for target in self.targets])
            self.current_target = self.targets[closest_cell_index]
            self.current_target_id = closest_cell_index
        else:
            self.current_target = self.end_target
        return 
    
    def _find_nearby_items(robot, item_map, world_grid, item_type, min_range=3,max_range=4, cache=None):
        nearby_items = []
        cell_x = int(robot.x // CELL_SIZE)
        cell_y = int(robot.y // CELL_SIZE)
        if (cell_x,cell_y) in cache:
            return cache[(cell_x,cell_y)]
        
        min_range_x = max(0, cell_x - min_range)
        max_range_x = min(MAP_DIMS[0], cell_x + max_range)
        min_range_y = max(0, cell_y - min_range)
        max_range_y = min(MAP_DIMS[1], cell_y + max_range)

        for i in range(min_range_x, max_range_x):
            for j in range(min_range_y, max_range_y):
                if item_map[i][j] == 1 and (type(world_grid[i][j][0]) == item_type):
                    nearby_items.extend(world_grid[i][j])
        cache[(cell_x,cell_y)] = nearby_items
        return nearby_items
    
    def _save_state(self, time):
        self.log_attitude.append((self.id, time,self.x,self.y,self.theta))
        self.log_rates.append((self.id, time,self.omega,self.vl,self.vr))
    
class BaseManualRobot(BaseRobot):
    def __init__(self, startpos,targets,end_target,robot_id, special=False):
        super().__init__(startpos,targets,end_target,robot_id, special)
        self.controller = ManualController()
      
    
    def _update_state(self, dt: float, wall_collided: pygame.Rect,event):
        if event:
            self.vl,self.vr = self.controller.calculate_motor_speed(event,self.vl,self.vr)
            
        self.avg_velocity = (self.vl + self.vr) / 2
        self.omega = (self.vl - self.vr) / self.size[0]
        self.theta += self.omega * dt
        if self.theta >= math.pi:
            self.theta -= 2 * math.pi
        if self.theta <= -math.pi:
            self.theta += 2 * math.pi 
        
 
        if self.collided:
            self._handle_collision(wall_collided)
        else:             
            cos_theta = math.cos(-self.theta)
            sin_theta = math.sin(-self.theta)
            self.x += self.avg_velocity * cos_theta * dt
            self.y -= self.avg_velocity * sin_theta * dt
            
        return
            
    
    def handler(self, dt, nearby_obstacles,event=None):
        self._handle_map_boundaries()
        self._check_task()
        wall_collided = self._get_collision(nearby_obstacles)
        self._update_state(dt, wall_collided, event)
    
class BaseAutoBot(BaseRobot):
    def __init__(self, startpos,targets,end_target,robot_id, special=False):
        super().__init__(startpos,targets,end_target,robot_id, special)
        self.sensor_spacing = [math.radians(-150),math.radians(-60),math.radians(0), math.radians(60), math.radians(150)]
        self.sensor = [0, 0, 0, 0, 0, 0]  # 1 tactile sensor (sensor[0]) and 5 ultrasound
        self.sensor_range = 45
        self.sensor_sweep = math.radians(25) # degrees
        # Precompute sensor angle ranges
        self.sensor_ranges = [(sensor_angle - self.sensor_sweep, sensor_angle + self.sensor_sweep) 
                        for sensor_angle in self.sensor_spacing]
        self.mental_map = np.zeros((MAP_DIMS[0],
                    MAP_DIMS[1]))
        self.visited_cells = set()
        
    def _update_mental_map(self, walls):
        if not self.added_walls:
            self.mental_map = self.mental_map - walls[:self.mental_map.shape[0],:self.mental_map.shape[1]]
            self.map_size = np.sum(self.mental_map == 0)
            self.added_walls = True
            
        cell_x = int(self.x // CELL_SIZE)
        cell_y = int(self.y // CELL_SIZE)
        self.visited_cells.add((cell_x,cell_y))
        self.mental_map[cell_x][cell_y] = 1
        self.completed_map = len(self.visited_cells)/self.map_size 
        return
    
    def _update_sensors(self, nearby_obstacles):
        # Reset sensors
        self.sensor = [0, 0, 0, 0, 0, 0]
        # Precompute cos and sin of theta
        rounded_theta = bin_angle(self.theta,1)
        cos_theta = math.cos(rounded_theta)
        sin_theta = math.sin(rounded_theta)
        
        for obstacle in nearby_obstacles:
            # Calculate distance to obstacle
            v2_x = obstacle.x // CELL_SIZE - self.x //CELL_SIZE
            v2_y = obstacle.y // CELL_SIZE - self.y // CELL_SIZE
            dist_to_wall = np.hypot(v2_x, v2_y)
            if dist_to_wall <= self.sensor_range:
                # Calculate the angle between robot direction and obstacle
                relative_angle = calc_angle((cos_theta, sin_theta), (v2_x, v2_y), cache=True)
                # Check which sensor should be activated
                for i, (sensor_start, sensor_end) in enumerate(self.sensor_ranges):
                    if sensor_start <= relative_angle < sensor_end:
                        self.sensor[i + 1] = (1 - dist_to_wall/self.sensor_range)
                        break
        return
    
    def _get_new_task(self):
        if not self.mission_complete:
            closest_cell_index = find_closest_cell((self.x // CELL_SIZE, self.y // CELL_SIZE), self.targets)
            self.current_target = self.targets[closest_cell_index]
            self.current_target_id = closest_cell_index
            self.shortest_route = round(calc_distance((self.x ,self.y), (next_target.x,next_target.y)))
        else:
            next_target = self.end_target
        return next_target
        
    
        
class BaseNeuroRobot(BaseAutoBot):
    def __init__(self, startpos,targets,end_target,robot_id, chromosome, n_hidden=N_HIDDEN, special=False):
        super().__init__(startpos,targets,end_target,robot_id, special)
        self.state = np.array([]).reshape(-1,1)
        self.error_to_goal =  0
        self.angle_w_goal = 0
        
class BaseForagingRobot(BaseRobot):
    def __init__(self, startpos,targets,end_target,robot_id=1, special=False):
        super().__init__(startpos,targets,end_target,robot_id, special)
        self.carrying = False
        self.carried_target_id = None
        self.targets_returned = 0
        self.base_range = self._get_base_range()

    def _get_base_range(self):
        start_pos_cell_x = self.startpos.x // CELL_SIZE
        start_pos_cell_y = self.startpos.y // CELL_SIZE
        min_range_x = -1
        max_range_x = 2
        min_range_y = -1
        max_range_y = 2
        base_range = []
        for i in range(min_range_x,max_range_x):
            for j in range(min_range_y,max_range_y):
                base_range.append((start_pos_cell_x + i, start_pos_cell_y + j))
        
        return base_range
    
    def _move_token(self, token):
        token.x = self.x
        token.y = self.y
        # token.rect.w = 40
        # token.rect.h = 40
        
    def _check_task(self):
        '''
        Function to check if we have collided with a token, when this happens the
        robot obtains a token
        '''
        target_reached = self.hitbox.collidelist(self.targets) 
        if target_reached == -1:
            return
         
        if self.current_target == self.targets[target_reached] and not self.carrying:
            self.carrying = True
            self.carried_target_id = target_reached
        
        if self.carrying:
            self._move_token(self.targets[self.carried_target_id])
            back_at_base = (self.x // CELL_SIZE,self.y // CELL_SIZE) in self.base_range
            if back_at_base:
                print("Got Target at location")
                self.targets.pop(self.carried_target_id)
                self.carrying = False
                self.targets_returned += 1
                self._get_new_task()
        # self.current_target = self._get_new_task()
        return 
        
class TankNeuroRobot(BaseNeuroRobot):
    def __init__(self, startpos,targets,end_target,robot_id, chromosome,n_hidden=N_HIDDEN, special=False):
        super().__init__(startpos,targets,end_target,robot_id,chromosome,n_hidden, special)
        self.tank_empty = False
        self.reached_end = False
        self.energy_in_tank = MAX_ENERGY  
        self.closeness = []
        self.time_active = None
        
        self.avg_velocity = (self.vl + self.vr) / 2
        self.state = np.array([self.avg_velocity, self.error_to_goal,self.angle_w_goal,self.theta,self.energy_in_tank,*self.sensor[1:]]).reshape(-1,1)
        self.controller = NeuroController(n_hidden=n_hidden, chromosome=chromosome)
    
    def _update_tank(self):
        if self.energy_in_tank <= 0:
            self.tank_empty = True
            self.img = self.tank_empty_img
            return
        kinetic_energy_used = 0.5 * self.mass * self.avg_velocity ** 2 
        passive_energy_used = MAX_ENERGY * 0.001
        total_energy_used = kinetic_energy_used + passive_energy_used
        self.energy_in_tank = max(self.energy_in_tank - total_energy_used, 0)
        return
    
    def get_reward(self):
        closeness = sum(self.closeness)
        fitness = round(
            W_TOKEN*self.targets_collected 
            + W_QUICK*(self.targets_collected/self.time_active)
            + closeness*W_CLOSE
            + self.collided_w_wall*W_COL
            + 1000*self.reached_end
            ,2)
        print(f"-------{self.id}--------")
        print(f"T: {self.targets_collected*W_TOKEN}, Q: {W_QUICK*(self.targets_collected/self.time_active)}, \
              Clo: {closeness*W_CLOSE}, Col: {self.collided_w_wall*W_COL}, Time: {(self.time_active)}")
        print(f"Fitness: {fitness}")
        return fitness
        
    def _get_collision(self, nearby_obstacles):
        """Function to calculate whetere collisions between the walls and the agent have occured"""
        NO_COLLISIONS = -1

        collided_walls = self.hitbox.collidelist(nearby_obstacles)
        if collided_walls != NO_COLLISIONS: # -1 equals no collision:
            self.collided = True
            self.sensor[0] = 1
            self.collided_w_wall -= 1
            self.energy_in_tank = max(self.energy_in_tank - MAX_ENERGY*0.01, 0)
            return nearby_obstacles[collided_walls]
        else:
            self.collided = False
            return 
    
    def _update_state(self, dt: float, wall_collided: pygame.Rect,time: float):
        
        if not self.tank_empty:
            self.time_active = time
            self.state = np.array([self.avg_velocity / MAX_SPEED, self.error_to_goal,self.angle_w_goal,self.omega/(2*math.pi),self.energy_in_tank / MAX_ENERGY ,*self.sensor[1:]]).reshape(-1,1)
            self.vl,self.vr = self.controller.calculate_motor_speed(self.state)
            self.avg_velocity = (self.vl + self.vr) / 2
            self.omega = (self.vl - self.vr) / self.moment_of_inertia
            self.theta += self.omega * dt
            
            if self.theta >= math.pi:
                self.theta -= 2 * math.pi
            if self.theta <= -math.pi:
                self.theta += 2 * math.pi
                
            if self.collided:
                self._handle_collision(wall_collided)
            else:             
                cos_theta = math.cos(-self.theta)
                sin_theta = math.sin(-self.theta)
                self.x += self.avg_velocity * cos_theta * dt
                self.y -= self.avg_velocity * sin_theta * dt
                
            rounded_theta = bin_angle(self.theta,0.1)
            if self.current_target:
                delta_x = self.current_target.x - self.x
                delta_y = self.current_target.y - self.y 
                r = np.hypot(delta_x,delta_y)
                
                cos_theta = math.cos(rounded_theta)
                sin_theta = math.sin(rounded_theta)
                self.angle_w_goal = calc_angle((cos_theta, sin_theta), (delta_x , delta_y )) / (math.pi)
                self.error_to_goal = 1 - bound(r / self.shortest_route,0,2)
                self.closeness.append(self.error_to_goal)
            else:
                self.angle_w_goal = -1
                self.error_to_goal = -1
        else:
            return
    
    def _check_task(self):
        '''
    Function to check if we have collided with a token, when this happens the
    robot obtains a token
    '''
        target_reached = self.hitbox.collidelist(self.targets) 
        if target_reached == -1:
            return 
        if self.current_target == self.targets[target_reached]:
            self.targets.pop(target_reached)
            self.targets_collected += 1
            self.energy_in_tank = min(self.energy_in_tank+MAX_ENERGY*0.3, MAX_ENERGY)
            if len(self.targets) == 0:
                self.mission_complete = True
                return 
            self.current_target = self._get_new_task()
        return 
    
            
    def handler(self, dt, nearby_obstacles, time):
        self._handle_map_boundaries()
        self._check_task()
        wall_collided = self._get_collision(nearby_obstacles)
        self._update_sensors(nearby_obstacles)
        self._update_tank()
        self._update_state(dt, wall_collided, time)
        self._save_state(time)

class SearchingTankNeuroRobot(TankNeuroRobot):
    def __init__(self, startpos,targets,end_target,robot_id, chromosome,n_hidden=N_HIDDEN, special=False):
        super().__init__(startpos,targets,end_target,robot_id,chromosome,n_hidden, special)
        self.searching = True
        self.current_view = self._get_current_view()
        self.added_walls = False
        self.map_size = 0
        
    def _update_tank(self):
        if self.energy_in_tank <= 0:
            return
        kinetic_energy_used = 0.5 * self.mass * self.avg_velocity ** 2 
        passive_energy_used = MAX_ENERGY * 0.001
        total_energy_used = kinetic_energy_used + passive_energy_used
        self.energy_in_tank = max(self.energy_in_tank - total_energy_used, 0)
        return
        
    def _get_new_task(self, nearby_tokens=None):
        if not self.mission_complete and nearby_tokens:
            closest_cell_index = find_closest_cell((self.x // CELL_SIZE, self.y // CELL_SIZE), nearby_tokens)
            next_target = nearby_tokens[closest_cell_index]
            self.shortest_route = round(calc_distance((self.x ,self.y), (next_target.x,next_target.y)))
            return next_target
        else:
            return None
        
    def _get_current_view(self):
        cell_x = self.x // CELL_SIZE
        cell_y = self.y // CELL_SIZE
        min_range_x = max(0, cell_x - 2)
        max_range_x = min(MAP_DIMS[0], cell_x + 3)
        min_range_y = max(0, cell_y - 2)
        max_range_y = min(MAP_DIMS[1], cell_y + 3)
        
        current_view = self.mental_map[min_range_x:max_range_x,min_range_y:max_range_y]
        
        return current_view
        
    def _check_task(self, nearby_tokens):    
        '''
    Function to check if we have collided with a token, when this happens the
    robot obtains a token
    '''
        if self.mission_complete:
            return
        
        if not nearby_tokens:
            self.searching = True
            self.current_target = None
            return
            
        if nearby_tokens and self.searching:    
            self.current_target = self._get_new_task(nearby_tokens)
            self.searching = False # A Target has been found 
            return
        
        elif nearby_tokens and not self.searching:
            target_reached = self.hitbox.collidelist(self.targets) 
            if target_reached == -1:
                return 
            else:
                self.targets.pop(target_reached)
                self.targets_collected += 1
                # self.energy_in_tank = min(self.energy_in_tank+MAX_ENERGY*0.3, MAX_ENERGY)
                self.searching = True
                self.current_target = None
                if len(self.targets) == 0:
                    self.mission_complete = True
                return   
            
        return 
    
    def _update_state(self, dt: float, wall_collided: pygame.Rect,time: float):
            
        # if not self.tank_empty:
        self.time_active = time
        self.state = np.array([self.avg_velocity / MAX_SPEED, self.error_to_goal,self.angle_w_goal,self.omega/(2*math.pi),self.energy_in_tank / MAX_ENERGY ,self.completed_map,*self.current_view.flatten(),*self.sensor[1:]]).reshape(-1,1)
        self.vl,self.vr = self.controller.calculate_motor_speed(self.state)
        self.avg_velocity = (self.vl + self.vr) / 2
        self.omega = (self.vl - self.vr) / self.moment_of_inertia
        self.theta += self.omega * dt
        
        if self.theta >= math.pi:
            self.theta -= 2 * math.pi
        if self.theta <= -math.pi:
            self.theta += 2 * math.pi
            
        if self.collided:
            self._handle_collision(wall_collided)
        else:             
            cos_theta = math.cos(-self.theta)
            sin_theta = math.sin(-self.theta)
            self.x += self.avg_velocity * cos_theta * dt
            self.y -= self.avg_velocity * sin_theta * dt
            
        rounded_theta = bin_angle(self.theta,0.1)
        if self.current_target:
            delta_x = self.current_target.x - self.x
            delta_y = self.current_target.y - self.y 
            r = np.hypot(delta_x,delta_y)
            
            cos_theta = math.cos(rounded_theta)
            sin_theta = math.sin(rounded_theta)
            self.angle_w_goal = calc_angle((cos_theta, sin_theta), (delta_x , delta_y )) / (math.pi)
            self.error_to_goal = 1 - bound(r / self.shortest_route,0,2)
            self.closeness.append(self.error_to_goal)
        else:
            self.angle_w_goal = -1
            self.error_to_goal = -1
        # else:
        #     return
        
    def get_reward(self):
        closeness = sum(self.closeness)
        fitness = round(
            10*self.targets_collected 
            # + closeness*W_CLOSE
            + self.collided_w_wall*0.1
            + len(self.visited_cells) * 0.3
            ,2)
        print(f"-------{self.id}--------")
        print(f"T: {self.targets_collected*10}, Q: {W_QUICK*(self.targets_collected/self.time_active)}, \
              Clo: {closeness*W_CLOSE}, Col: {self.collided_w_wall*0.1}, Time: {(self.time_active)}, Vis: {(len(self.visited_cells) ) * 0.3}")
        print(f"Fitness: {fitness}")
        return fitness
    
   
    def handler(self, dt, world_map:WorldMap, time):
        self._handle_map_boundaries()
        
        nearby_obstacles = self.find_nearby_items(world_map.wall_map,world_map.spatial_grid)
        token_map = build_token_map(self.targets)
        nearby_tokens = self.find_nearby_items(token_map,world_map.spatial_grid, min_range=6,max_range=7)
        self._check_task(nearby_tokens)
        wall_collided = self._get_collision(nearby_obstacles)
        self._update_sensors(nearby_obstacles)
        self._update_mental_map(world_map.wall_map)
        self._update_state(dt, wall_collided, time)
        self._save_state(time)
        
class DebugBot(BaseManualRobot, BaseAutoBot):
    def __init__(self,startpos, targets,end_target, robot_id=1, special=False):
        super().__init__(startpos,targets,end_target,robot_id, special)
        
    def handler(self, dt, nearby_obstacles,event=None):
        self._handle_map_boundaries()
        self._check_task()
        wall_collided = self._get_collision(nearby_obstacles)
        self._update_sensors(nearby_obstacles)
        self._update_state(dt, wall_collided, event)
        

class ForagingManualRobot(BaseManualRobot, BaseForagingRobot):
    def __init__(self,startpos, targets,end_target, robot_id=1, special=False):
        super().__init__(startpos,targets,end_target,robot_id, special)
        self.controller = ManualController()
        self._get_new_task()
        
    
    def _check_task(self):
        '''
        Function to check if we have collided with a token, when this happens the
        robot obtains a token
        '''
        
        target_reached = self.hitbox.collidelist([target.rect for target in self.targets]) 
        if target_reached == -1:
            return
        
        if self.current_target == self.targets[target_reached] and not self.carrying:
            self.targets[target_reached].being_carried = True
            self.carrying = True
            self.carried_target_id = target_reached
            self.current_target = self.startpos
        
        if self.carrying:
            self._move_token(self.targets[self.carried_target_id])
            back_at_base = (self.x // CELL_SIZE,self.y // CELL_SIZE) in  self.base_range
            if back_at_base:
                print("Got Target at location")
                self.targets.pop(self.carried_target_id)
                self.carrying = False
                self.targets_returned += 1
                self._get_new_task()
        return 

    def handler(self, dt, nearby_obstacles,event=None):
        self._handle_map_boundaries()
        wall_collided = self._get_collision(nearby_obstacles)
        self._check_task()
        self._update_state(dt, wall_collided, event)
        
        
class ForagingNeuroBot(BaseNeuroRobot, BaseForagingRobot):
        def __init__(self, startpos,targets,end_target,robot_id, chromosome,n_hidden=N_HIDDEN, special=False):
            super().__init__(startpos,targets,end_target,robot_id,chromosome,n_hidden, special)
            self.controller = NeuroController(n_hidden=n_hidden, chromosome=chromosome)
            self.current_target_id = None
            self.searching = True
            
        def _check_task(self, nearby_tokens):
            '''
            Function to check if we have collided with a token, when this happens the
            robot obtains a token
            '''
            
            # If robot has not yet picked a spot to go and tokens are in the area 
            if nearby_tokens and self.searching and not self.carrying:    
                self._get_new_task(nearby_tokens)
                self.searching = False # A Target has been found 
                return
            
            # If target has already been picked up by another robot, this one needs to change its task            
            if self.targets[self.current_target_id].being_carried and not self.carrying:
                self._get_new_task(nearby_tokens)
                self.searching = True
                return
            
            # Reset if there are no tokens in the area or robot drifted too far 
            if (not nearby_tokens and not self.carrying) \
                or (abs(self.error_to_goal) >= 2 and not self.carrying):
                self.searching = True
                self._get_new_task(nearby_tokens)
                return
        
            if nearby_tokens and not self.searching:
                target_reached = self.hitbox.collidelist([target.hitbox for target in self.targets]) 
                if target_reached == -1:
                    return
                
                if self.current_target == self.targets[target_reached] and not self.carrying:
                    self.targets_collected += 1
                    self.targets[target_reached].being_carried = True
                    self.carrying = True
                    self.carried_target_id = target_reached
                    self.current_target = self.startpos
                    return
                
            if self.carrying:
                self._move_token(self.targets[self.carried_target_id])
                back_at_base = (self.x // CELL_SIZE,self.y // CELL_SIZE) in self.base_range
                if back_at_base:
                    print("Got Target at location")
                    self.targets.pop(self.carried_target_id)
                    self.carrying = False
                    self.targets_returned += 1
                    self._get_new_task(nearby_tokens)
                    return
                return 
            
        
        def _get_new_task(self, nearby_tokens):
            
            if not self.mission_complete:
                tokens_not_being_carried = [target for target in nearby_tokens if not target.being_carried]
                closest_cell_index = find_closest_cell((self.x // CELL_SIZE, self.y // CELL_SIZE), [target.hitbox for target in tokens_not_being_carried])
                if type(closest_cell_index) == int:
                    self.current_target = tokens_not_being_carried[closest_cell_index]
                    self.current_target_id = find_tile_id(self.targets, nearby_tokens[closest_cell_index])
                    self.shortest_route = round(calc_distance((self.x ,self.y), (self.current_target.x,self.current_target.y)))
                else:
                    self.current_target = None
                    self.current_target_id = -1
            else:
                self.current_target = None
                self.current_target_id = -1
            return 
        
        def _update_state(self, dt: float, wall_collided: pygame.Rect,time: float):
            self.time_active = time
            self.state = np.array([self.avg_velocity / MAX_SPEED, self.error_to_goal,self.angle_w_goal,self.omega/(2*math.pi),*self.sensor[1:]]).reshape(-1,1)
            self.vl,self.vr = self.controller.calculate_motor_speed(self.state)
            self.avg_velocity = (self.vl + self.vr) / 2
            self.omega = (self.vl - self.vr) / self.moment_of_inertia
            self.theta += self.omega * dt
            
            if self.theta >= math.pi:
                self.theta -= 2 * math.pi
            if self.theta <= -math.pi:
                self.theta += 2 * math.pi
                
            if self.collided:
                self._handle_collision(wall_collided)
            else:             
                cos_theta = math.cos(-self.theta)
                sin_theta = math.sin(-self.theta)
                self.x += self.avg_velocity * cos_theta * dt
                self.y -= self.avg_velocity * sin_theta * dt
                
            rounded_theta = bin_angle(self.theta,0.1)
            if self.current_target:
                delta_x = self.current_target.x - self.x
                delta_y = self.current_target.y - self.y 
                r = np.hypot(delta_x,delta_y)
                
                cos_theta = math.cos(rounded_theta)
                sin_theta = math.sin(rounded_theta)
                self.angle_w_goal = calc_angle((cos_theta, sin_theta), (delta_x , delta_y )) / (math.pi)
                self.error_to_goal = 1 - bound(r / self.shortest_route,0,2)
            else:
                self.angle_w_goal = -1
                self.error_to_goal = -1
                
        def get_reward(self):
     
            targets_returned = 25 * self.targets_returned
            targets_collected = 10 * self.targets_collected
            collided = 0.1*self.collided_w_wall
            fitness = round(
            targets_returned
            # + closeness*W_CLOSE
            + collided
            + targets_collected
            ,2)
            print(f"-------{self.id}--------")
            print(f"Tr: {targets_returned}, Tc: {targets_collected}, Col: {collided}")
            print(f"Fitness: {fitness}")
            return fitness
        
        def handler(self, dt, world_map:WorldMap, time):
            self._handle_map_boundaries()
            
            nearby_obstacles = self._find_nearby_items(world_map.wall_map,world_map.spatial_grid, cache=wall_cache, item_type=Tile)
            token_map = build_token_map([target.hitbox for target in self.targets])
            nearby_tokens = self._find_nearby_items(token_map,world_map.spatial_grid, min_range=6,max_range=7, cache=token_cache, item_type=Target)
            self._check_task(nearby_tokens)
            wall_collided = self._get_collision(nearby_obstacles)
            self._update_sensors(nearby_obstacles)
            # self._update_mental_map(world_map.wall_map)
            self._update_state(dt, wall_collided, time)
            self._save_state(time)


        
        