from wheeled_robot_class import GeneticAlgorithm
import pygame
import matplotlib.pyplot as plt

n_iter = 10
ls_fitness = list()
ls_cross_rate = list()
pygame.init()
cross_rate = 0.8
GA = GeneticAlgorithm(n_robots=2 * 6, n_iter=n_iter, cross_rate=cross_rate,
                      mut_rate=1 / 12)  # n_robots should be an equal number for crossover to work





GA.main(sim_time=15000)

pygame.quit()

x = range(0, n_iter + 1)
plt.figure()
plt.grid()
plt.plot(x, GA.get_results('fitness'))

plt.figure()
plt.grid()
plt.plot(x, GA.get_results('rel_dist'))
