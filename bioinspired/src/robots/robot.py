import numpy as np
import pygame
import math
import pygame.locals

from src.controllers.controllers import BaseController

from ..utility.functions import calc_angle, calc_distance, bin_angle, find_closest_cell, bound
from ..world_map.world_map import WorldMap
from ..utility.constants import *


class Robot:
    def __init__(self,  startpos,endpos, controller:BaseController, token_locations, special_flag,robot_id=0):
        self.id = robot_id + 1
        self.controller = controller
        self.w = ROBOT_WIDTH * 20
        self.tokens_locations = token_locations
        self.init_pos = startpos
        self.x = startpos[0]
        self.y = startpos[1]
        self.endpos = endpos
        self.vx = 0
        self.vy = 0
        self.current_cell = (self.x // CELL_SIZE, self.y // CELL_SIZE)
        self.delta_x = 0 # difference between current x and desired x
        self.delta_y = 0 # difference between current y and desired y
        self.error_to_goal = 0
        self.theta = 0

        self.energy_tank = MAX_ENERGY
        self.tank_empty = False
        self.time_active = 0

        self.found_all_tokens = False
        self.reached_end = False
        
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
        self.closeness = []
        self.ls_diff_pos_error = []

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
        
        theta = round(self.theta/math.pi,2)
        energy = self.energy_tank / MAX_ENERGY
        self.angle_w_next_token = 0
        self.state = np.array([self.vl,self.vr,self.error_to_goal,self.angle_w_next_token,theta,energy,*self.sensor[1:]]).reshape(-1,1)

        self.width = ROBOT_WIDTH 
        self.length = ROBOT_WIDTH 
        self.collision = 0
        self.collided = False
        self.special = special_flag
        self.token = 0
        self.next_token = []
        self.time_stamps = list()
        self.need_next_token = True
        self.same_cell = 0


    def _attitude(self):
        return(self.x,self.y,math.degrees(self.theta))
    

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
                relative_angle = calc_angle((cos_theta, sin_theta), (v2_x, v2_y), cache=True)
                # Check which sensor should be activated
                for i, (sensor_start, sensor_end) in enumerate(self.sensor_ranges):
                    if sensor_start <= relative_angle < sensor_end:
                        self.sensor[i + 1] = (1 - dist_to_wall/self.sensor_range)
                        break
        
        return

        
    def get_reference_position(self):
            
        if self.need_next_token:
            if not self.found_all_tokens:
                closest_cell_index = find_closest_cell((self.x // CELL_SIZE, self.y // CELL_SIZE), self.tokens_locations)
                self.next_token = self.tokens_locations[closest_cell_index]
            else:
                self.next_token = self.endpos
            self.shortest_route = round(calc_distance((self.x ,self.y), (self.next_token.x,self.next_token.y)))
            self.need_next_token = False
        
        if not self.tank_empty:
            self.delta_x = self.next_token.x - self.x
            self.delta_y = self.next_token.y - self.y 
            r = np.hypot(self.delta_x,self.delta_y)
            rounded_theta = bin_angle(self.theta,0.1)
            cos_theta = math.cos(rounded_theta)
            sin_theta = math.sin(rounded_theta)
            self.angle_w_next_token = calc_angle((cos_theta, sin_theta), (self.delta_x , self.delta_y )) / (math.pi)
            self.error_to_goal = 1 - bound(r / self.shortest_route,0,2)
            self.closeness.append(self.error_to_goal)

        if self.next_token not in self.tokens_locations:
            self.need_next_token = True
        
        
    def update_state(self, t=None):
        if not self.reached_end and not self.tank_empty:
            self.time_active = t
        if self.energy_tank <= 0.0 and not self.tank_empty:
            self.tank_empty = True
        self.get_reference_position()
        omega = round(self.omega/(2*math.pi),2)
        energy = 1 - (self.energy_tank / MAX_ENERGY)
        vr = self.vr/self.maxspeed
        vl = self.vl/self.maxspeed
        self.state = np.array([vl,vr,self.error_to_goal,self.angle_w_next_token,omega,energy,*self.sensor[1:]]).reshape(-1,1)
        return
        
    def get_tokens(self, t):
        '''
        Function to check if we have collided with a token, when this happens the
        robot obtains a token
        '''
        if len(self.tokens_locations) == 0:
            # Found all Tokens
            self.found_all_tokens = True
            self.get_end_tile()
            return self.endpos
        token_collected = self.hitbox.collidelist(self.tokens_locations) 
        if token_collected == -1:
            return self.next_token
        if self.next_token == self.tokens_locations[token_collected]:
            self.token += 1
            self.energy_tank = min(self.energy_tank+MAX_ENERGY*0.3, MAX_ENERGY)
            self.time_stamps.append(t)
            self.tokens_locations.pop(token_collected)
        return self.next_token

    def get_end_tile(self):
        found_end_tile = self.hitbox.collidelist([self.endpos])
        if found_end_tile == -1:
            return
        else:
            self.reached_end = True 


    def move(self, wall_collided, dt, event=None, auto=False):
        if self.tank_empty or self.reached_end:
            return
        
        self.ls_x.append(self.x)
        self.ls_y.append(self.y)
        self.ls_theta.append(self.theta)
        if len(self.ls_x) >= 2:
            delta_x = self.ls_x[-2] - self.ls_x[-1]
            delta_y = self.ls_y[-2] - self.ls_y[-1]
            delta_r = np.hypot(delta_x, delta_y)  # Efficient calculation of sqrt(delta_x^2 + delta_y^2)
            self.dist_travelled += delta_r
        
        self.vl,self.vr = self.controller.calculate_motor_speed(self.state)
        
        kinetic_energy_used = 0.5*1e-5*np.hypot(self.vl,self.vr)**2 
        passive_energy_used = MAX_ENERGY*0.001
        total_energy_used = kinetic_energy_used + passive_energy_used
        self.energy_tank = max(self.energy_tank-total_energy_used, 0)
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
        quicknes = []
        closeness = sum(self.closeness)
        if self.time_stamps:
            time_between_tokens = [self.time_stamps[0]]
            for i, _ in enumerate(self.time_stamps[1:]):
                time_between_tokens.append(abs(self.time_stamps[i] - self.time_stamps[i-1]))
            quicknes = [1/time for time in time_between_tokens]


        fitness = round(
            W_TOKEN*self.token 
            + W_QUICK*(self.token/self.time_active)
            + closeness*W_CLOSE
            + self.collision*W_COL
            + 1000*self.reached_end
            ,2)
        print(f"-------{self.id}--------")
        print(f"T: {self.token*W_TOKEN}, Q: {W_QUICK*(self.token/self.time_active)}, \
              Clo: {closeness*W_CLOSE}, Col: {self.collision*W_COL}, Time: {(self.time_active)}")
        print(f"Fitness: {fitness}")
        return fitness
    



        