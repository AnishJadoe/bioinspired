import sys
import numpy as np
import math
import pickle
import os
import functools

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

def get_init_chromosomes_heuristic(n_robots):
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

def get_init_chromosomes_NN(n_robots, n_inputs, n_outputs, n_hidden, seed=42):
    """
    Initializes a population of chromosomes for a neural network.

    Parameters:
        n_robots (int): Number of robots (population size).
        n_inputs (int): Number of input neurons.
        n_outputs (int): Number of output neurons.
        n_hidden (int): Number of hidden neurons.
        seed (int, optional): Seed for the random number generator. Default is 42.

    Returns:
        list: List of chromosomes, where each chromosome is a numpy array containing the weights and biases.
    """
    np.random.seed(seed)
    population = []
    
    n_weights = n_inputs * n_hidden + n_hidden * n_outputs
    n_biases = n_hidden + n_outputs
    
    # Using uniform distribution in the range [-0.5, 0.5]
    weight_range = 0.5
    
    for _ in range(n_robots):
        weights = np.random.uniform(low=-weight_range, high=weight_range, size=n_weights)
        biases = np.random.uniform(low=-weight_range, high=weight_range, size=n_biases)
        chromosome = np.concatenate((weights, biases))
        population.append(chromosome)
        
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

angle_cache = {}
def calc_angle(v1, v2, cache=False):
    """Calculate the orientation of 2 points with respect to each other.

    Args:
        v1 (tuple or list of float): First vector (x, y).
        v2 (tuple or list of float): Second vector (x, y).

    Returns:
        float: The orientation angle in radians.
    """
 
    if (v1, v2) in angle_cache and cache:
        return angle_cache[(v1, v2)]
    
    # Normalize both direction vectors
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    
    # Calculate the dot product and the angle
    dot_product = np.dot(v1_normalized, v2_normalized)
    dot_product = max(min(dot_product, 1.0), -1.0)
    angle = math.acos(dot_product)
    
    # Calculate the cross product to determine the sign
    cross_product = np.cross(v1_normalized, v2_normalized)
    
    # Determine the sign of the angle
    if cross_product < 0:
        angle *= -1
    if cache:
        angle_cache[(v1, v2)] = angle
    return angle

def cache_size():
    """Returns the size of the angle_cache."""
    return len(angle_cache)

def get_size(obj, seen=None):
    """Recursively finds the size of objects in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def cache_size_kb():
    """Returns the size of the angle_cache in kilobytes."""
    size_bytes = get_size(angle_cache)
    size_kb = size_bytes / 1024
    return size_kb