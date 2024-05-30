import pickle
import os
from results.plot import plot_mean_results, plot_best_results


def load_GAs(GAs, folder):
    for file in os.scandir(folder):
        if os.path.isdir(file):
            load_GAs(GAs, file)
        else:
            with open(file, "rb") as f:
                GA = pickle.load(f)
                GAs.append(GA)
    return GAs


FOLDER = r"saved_runs\n_robots_250\epochs_30\epoch_time_60000"
GAs = []
GAs = load_GAs(GAs,FOLDER)

plot_mean_results(GAs)
