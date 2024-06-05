from src.utility.functions import load_GAs
from results.plot import plot_mean_results, plot_best_results, plot_aggregate_results


FOLDER = r"saved_runs\n_robots_150\epochs_20"
GAs = []
GAs = load_GAs(GAs,FOLDER)

plot_mean_results(GAs)
plot_best_results(GAs)