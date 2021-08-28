# Loading in functions
from genetic_algorithm import GeneticAlgorithm
from walls import Walls
from robot import Robot
from environment import Envir
from draw import Draw
import pygame
import matplotlib.pyplot as plt
import numpy as np



epochs = 20
GA = GeneticAlgorithm(n_robots=2 * 15, n_iter=epochs, cross_rate=0.8,
                      mut_rate=1 / 5)  # n_robots should be an equal number for crossover to work

best, best_eval = GA.main(60000)

x = range(0, epochs + 1)

plt.figure()
plt.title('FITNESS')
plt.grid()
plt.plot(x, GA.get_results('fitness'))

plt.figure()
plt.title('TOT DIST')
plt.grid()
plt.plot(x, GA.get_results('tot_dist'))

plt.figure()
plt.title('TOT COLL')
plt.grid()
plt.plot(x, GA.get_results('coll'))

plt.figure()
plt.title('REL DIST')
plt.grid()
plt.plot(x, GA.get_results('rel_dist'))

plt.figure()
plt.title('TOKEN')
plt.grid()
plt.plot(x, GA.get_results('token'))

plt.show()
print(best, best_eval)
