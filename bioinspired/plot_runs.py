from src.utility.functions import load_GAs
from results.plot import plot_mean_results, plot_best_results, plot_aggregate_results


FOLDER =r"saved_runs\n_robots_100\epochs_250\epoch_time_30000\seed_42\neurons_5"
GAs = []
GAs = load_GAs(GAs,FOLDER)

plot_best_results(GAs)
plot_mean_results(GAs)