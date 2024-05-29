import numpy as np
import math

################ FUNCTIONS ################


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
        population.append(np.random.randint(low=-255, high=255, size=(32, 2)))
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


def calc_angle(coord1, coord2):
    """Calculate the orientation of 2 points with respect to eachother

    Args:
        coord1 (int)
        coord2 (int)

    Returns:
        orient (float)
    """
    if coord1[0] == coord2[0]:
        # If the points are at the same location there is no angle
        orient = 0
    else:
        orient = math.tan((coord1[1] - coord2[1]) / (coord1[0] - coord2[0]))

    return orient
