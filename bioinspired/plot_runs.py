from matplotlib import pyplot as plt
from src.utility.functions import load_GAs
from results.plot import plot_mean_results, plot_best_results, plot_aggregate_results, plot_best_fitness


FOLDER =r"saved_runs/n_robots_60/epochs_40"
GAs = []
GAs = load_GAs(GAs,FOLDER)

plot_best_fitness(GAs)
plt.show()