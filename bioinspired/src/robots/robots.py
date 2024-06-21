import math

import pygame
import numpy as np
from src.utility.functions import bin_angle, bound, calc_angle, calc_distance, find_closest_cell
from src.world_map.world_map import WorldMap
from ..utility.constants import *
from ..controllers.controllers import ManualController, NeuroController

ROBOT_WIDTH = 20
MAX_SPEED = 255
class BaseRobot():
    def __init__(self, startpos, targets,end_target, robot_id, special=False):
        self.id = robot_id
        self.x = startpos[0]
        self.y = startpos[1]
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
        
        self.current_target = self._get_new_task()
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

        collided_walls = self.hitbox.collidelist(nearby_obstacles)
        if collided_walls != NO_COLLISIONS: # -1 equals no collision:
            self.collided = True
            return nearby_obstacles[collided_walls]
        else:
            self.collided = False
            return 
    
    def _update_state(self, dt: float, wall_collided:pygame.rect.Rect):
        raise NotImplementedError

    
    def _handle_collision(self, wall_collided: pygame.rect.Rect):
        wall_elasticity = 1
        if (self.x - self.size[0] / 2) > wall_collided.x and (self.y  - self.size[1] / 2) > wall_collided.y:
            self.x += wall_elasticity + 1
            self.y += wall_elasticity
        elif (self.x + self.size[0] / 2) <= wall_collided.x and (self.y - self.size[1] / 2) >= wall_collided.y:
            self.x -= wall_elasticity
            self.y += wall_elasticity + 1
        elif (self.x - self.size[0] / 2) >= wall_collided.x and (self.y + self.size[1] / 2) <= wall_collided.y:
            self.x += wall_elasticity
            self.y -= wall_elasticity + 1
        elif (self.x + self.size[0] / 2)  <= wall_collided.x and (self.y + self.size[1] / 2) <= wall_collided.y:
            self.x -= wall_elasticity + 1
            self.y -= wall_elasticity
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
            closest_cell_index = find_closest_cell((self.x // CELL_SIZE, self.y // CELL_SIZE), self.targets)
            next_target = self.targets[closest_cell_index]
        else:
            next_target = self.end_target
        return next_target
    
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
    
    
        
class BaseNeuroRobot(BaseAutoBot):
    def __init__(self, startpos,targets,end_target,robot_id, chromosome, n_hidden=N_HIDDEN, special=False):
        super().__init__(startpos,targets,end_target,robot_id, special)
        self.state = np.array([]).reshape(-1,1)
        self.error_to_goal =  0
        self.angle_w_next_token = 0
        

class TankNeuroRobot(BaseNeuroRobot):
    def __init__(self, startpos,targets,end_target,robot_id, chromosome,n_hidden=N_HIDDEN, special=False):
        super().__init__(startpos,targets,end_target,robot_id,chromosome,n_hidden, special)
        self.tank_empty = False
        self.reached_end = False
        self.energy_in_tank = MAX_ENERGY  
        self.closeness = []
        self.time_active = None
        
        self.avg_velocity = (self.vl + self.vr) / 2
        self.state = np.array([self.avg_velocity, self.error_to_goal,self.angle_w_next_token,self.theta,self.energy_in_tank,*self.sensor[1:]]).reshape(-1,1)
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
            self.energy_in_tank = max(self.energy_in_tank - MAX_ENERGY*0.005, 0)
            return nearby_obstacles[collided_walls]
        else:
            self.collided = False
            return 
    
    def _get_new_task(self):
        if not self.mission_complete:
            closest_cell_index = find_closest_cell((self.x // CELL_SIZE, self.y // CELL_SIZE), self.targets)
            next_target = self.targets[closest_cell_index]
            self.shortest_route = round(calc_distance((self.x ,self.y), (next_target.x,next_target.y)))
        else:
            next_target = self.end_target
        return next_target
    
    def _update_state(self, dt: float, wall_collided: pygame.Rect,time: float):
        
        if not self.tank_empty:
            self.time_active = time
            self.state = np.array([self.avg_velocity / MAX_SPEED, self.error_to_goal,self.angle_w_next_token,self.omega/(2*math.pi),self.energy_in_tank / MAX_ENERGY ,*self.sensor[1:]]).reshape(-1,1)
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
            delta_x = self.current_target.x - self.x
            delta_y = self.current_target.y - self.y 
            r = np.hypot(delta_x,delta_y)
         
            cos_theta = math.cos(rounded_theta)
            sin_theta = math.sin(rounded_theta)
            self.angle_w_next_token = calc_angle((cos_theta, sin_theta), (delta_x , delta_y )) / (math.pi)
            self.error_to_goal = 1 - bound(r / self.shortest_route,0,2)
            self.closeness.append(self.error_to_goal)
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

class DebugBot(BaseManualRobot, BaseAutoBot):
    def __init__(self,startpos, targets,end_target, robot_id=1, special=False):
        super().__init__(startpos,targets,end_target,robot_id, special)
        
    def handler(self, dt, nearby_obstacles,event=None):
        self._handle_map_boundaries()
        self._check_task()
        wall_collided = self._get_collision(nearby_obstacles)
        self._update_sensors(nearby_obstacles)
        self._update_state(dt, wall_collided, event)
        

    
        