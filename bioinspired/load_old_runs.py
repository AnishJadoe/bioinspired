import pickle
from results.plot import plot_mean_results, plot_best_results
with open("20pop_0.8mut_1cross_1000sim_2.pkl", "rb") as f:
    GA = pickle.load(f)

plot_best_results(GA)
plot_mean_results(GA)
