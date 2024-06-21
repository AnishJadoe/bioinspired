
import pygame
import math
import pygame.locals

from src.robots.robots import BaseRobot

from ..utility.constants import *

def draw_tank_percentage(robot, world):
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Tank: {round((robot.energy_in_tank/MAX_ENERGY) * 100),1}%', True, (0,0,0))
    world.blit(img,(robot.x,robot.y))

def draw_sensor_orientation(robot,world):
    for i,sensor in enumerate(robot.sensor[1:]):
        sensor_angle = robot.sensor_spacing[i]
        pygame.draw.line(world, SENSOR_COLORS[i], (robot.x, robot.y), 
        (robot.x+math.cos(robot.theta+sensor_angle)*robot.sensor_range,
        robot.y+math.sin(robot.theta+sensor_angle)*robot.sensor_range),width=2)

        pygame.draw.line(world, SENSOR_COLORS[i], (robot.x, robot.y), 
        (robot.x+math.cos(robot.theta+sensor_angle + robot.sensor_sweep)*robot.sensor_range,
        robot.y+math.sin(robot.theta+sensor_angle + robot.sensor_sweep)*robot.sensor_range),width=2)

        pygame.draw.line(world, SENSOR_COLORS[i], (robot.x, robot.y), 
        (robot.x+math.cos(robot.theta+sensor_angle - robot.sensor_sweep)*robot.sensor_range,
        robot.y+math.sin(robot.theta+sensor_angle - robot.sensor_sweep)*robot.sensor_range),width=2)

def draw_sensor_activation(robot,world):
    for i,sensor in enumerate(robot.sensor[1:]):
        if sensor:
            sensor_angle = robot.sensor_spacing[i]
            pygame.draw.line(world, SENSOR_COLORS[i], (robot.x, robot.y), 
            (robot.x+math.cos(robot.theta+sensor_angle)*robot.sensor_range,
            robot.y+math.sin(robot.theta+sensor_angle)*robot.sensor_range),width=6)

            pygame.draw.line(world, SENSOR_COLORS[i], (robot.x, robot.y), 
            (robot.x+math.cos(robot.theta+sensor_angle + robot.sensor_sweep)*robot.sensor_range,
            robot.y+math.sin(robot.theta+sensor_angle + robot.sensor_sweep)*robot.sensor_range),width=6)

            pygame.draw.line(world, SENSOR_COLORS[i], (robot.x, robot.y), 
            (robot.x+math.cos(robot.theta+sensor_angle - robot.sensor_sweep)*robot.sensor_range,
            robot.y+math.sin(robot.theta+sensor_angle - robot.sensor_sweep)*robot.sensor_range),width=6)

def draw_motor_speed(robot,world):
    font = pygame.font.SysFont(None, 20)
    img = font.render(f'vr: {round(robot.vr,2)}' , True, (0,0,0))
    world.blit(img,(robot.x+20,robot.y-10))
    img = font.render(f'vl: {round(robot.vl,2)}', True, (0,0,0))
    world.blit(img,(robot.x-50,robot.y-10))

def debug_theta(robot,world):
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'theta: {round(math.degrees(robot.theta),1)}', True, (0,0,0))
    world.blit(img,(robot.x,robot.y))
    pygame.draw.line(world, RED, (robot.x, robot.y), 
    (robot.x+math.cos(robot.theta)*robot.sensor_range,
    robot.y+math.sin(robot.theta)*robot.sensor_range),width=2)


def debug_token(robot,world):
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'tokens: {robot.token}', True, (0,0,0))
    world.blit(img,(robot.x,robot.y))

def draw_token(robot,world):
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Tokens collected: {robot.token}', True, (0,0,0))
    world.blit(img,(500,100))    

def draw_visited_cells(robot,world):
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Visited Cells: {len(robot.visited_cells)}', True, (0,0,0))
    world.blit(img,(robot.x,robot.y))  

def draw_robot(robot : BaseRobot,world:pygame.surface.Surface, debug=False):
    img = robot.img
    robot.trans_img = pygame.transform.rotozoom(img,
                                                math.degrees(-robot.theta), 1)
    robot.hitbox = robot.trans_img.get_rect(center=(robot.x, robot.y))
    world.blit(robot.trans_img, robot.hitbox)

def draw_robot_id(robot,world):
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'ID: {robot.id}', True, (0,0,0))
    world.blit(img,(robot.x,robot.y-20))  

def draw_error_to_goal(robot,world):
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Error: {round(robot.error_to_goal,2)}', True, (0,0,0))
    world.blit(img,(robot.x,robot.y-20)) 
    
def draw_angle_to_next_token(robot,world):
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Angle: {round(math.degrees(robot.angle_w_next_token),2)}', True, (0,0,0))
    world.blit(img,(robot.x,robot.y-20))
     
def draw_next_token(token,world):
    pygame.draw.rect(world,BLUE,token)
    
def draw_end_pos(endpos,world):
    pygame.draw.rect(world,ORANGE,endpos)
    
def draw_robot_bb(robot,world):
    pygame.draw.rect(world,RED,robot.hitbox, width=1)


# def draw_next_token(robot,world):
#     pygame.draw.line(world,BLUE, (robot.x,robot.y), (robot.next_token.x,robot.next_token.y))

def draw(robot, world):
    '''
    Draws on the world
    '''
    draw_robot(robot, world)
    # robot.draw_angle_to_next_token(world)
    # robot.draw_next_token(world)
    # robot.debug_theta(world)
    # robot.draw_tank_percentage(world)
    # robot.draw_error_to_goal(world)
    # robot.draw_next_token(world)
    # robot.draw_visited_cells(world)
    # robot.draw_robot_bb(world)
    #robot.debug_token(world)
    # robot.draw_sensor_orientation(world)
    # robot.draw_sensor_activation(world)
    # robot.debug_theta(world)
    # robot.draw_token(world)
    
def draw_time(world, time):
    font = pygame.font.SysFont(None, 36)
    txt = font.render(f'Time: {round(time,1)}', True, BLUE)
    world.blit(txt,(750,50))

def draw_gen(world, gen):
    font = pygame.font.SysFont(None, 24)
    txt = font.render(f'Gen: {gen}', True, BLUE)
    world.blit(txt,(800,75))