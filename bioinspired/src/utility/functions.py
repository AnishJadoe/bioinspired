import numpy as np
import math
import pickle
import os

################ FUNCTIONS ################

def load_GAs(GAs, folder):
    for file in os.scandir(folder):
        if os.path.isdir(file):
            load_GAs(GAs, file)
        else:
            with open(file, "rb") as f:
                GA = pickle.load(f)
                GAs.append(GA)
    return GAs

def get_init_pop(n_robots):
    """Generate the initial population to run the algorithm with

    Args:
        n_robots (int): The amount of robots in the population

    Returns:
        population (list): The initial population
    """
    np.random.RandomState()
    population = list()
    np.random.seed(42)  # To ensure that the initial population stays the same
    # with differing mutation parameters

    for i in range(0, n_robots):
        population.append(np.random.randint(low=-255, high=255, size=(64, 2)))
    return population


def calc_distance(coord1, coord2):
    """Calculates the distance between 2 points

    Args:
        coord1 (int)
        coord2 (int)

    Returns:
        dist (float)
    """
    dist = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    return dist


def calc_angle(v1, v2):
    """Calculate the orientation of 2 points with respect to eachother

    Args:
        coord1 (int)
        coord2 (int)

    Returns:
        orient (float)
    """

    # normalize both direction vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    orient = math.acos(np.dot(v1,v2))
    sign = np.cross(v1,v2)
    if sign < 0:
        orient *= -1
    if sign > 0:
            orient *= 1
    if sign == 0:
        if sign == -1:
            orient = math.pi
        else:
            orient = 0


    return orient
