import numpy as np 
import pygame 
from environment import Envir
from walls import Walls
from draw import Draw
import math



################ FUNCTIONS ################

def get_init_pop(n_robots):
    population = list()
    for i in range(0, n_robots):
        population.append(np.random.randint(low=-255, high=255, size=(10, 2)))
    return population


def calc_distance(coord1, coord2):
    dist = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    return dist


def calc_angle(coord1, coord2):
    if coord1[0] == coord2[0]:
        orient = 0
    else:
        orient = math.tan((coord1[1] - coord2[1]) / (coord1[0] - coord2[0]))

    return orient

