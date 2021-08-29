# Loading in functions
from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt


plt.style.use("bmh")

epochs = 10
x = range(0, epochs + 1)
ls_cross_rate = [0.1, 0.5, 0.8, 1]

fig_fitness, ax_fitness = plt.subplots()
ax_fitness.set(
    xlabel="Generation", ylabel="Fitness", title="Average Fitness Per Generation"
)

fig_rel_distance, ax_rel_distance = plt.subplots()
ax_rel_distance.set(
    xlabel="Generation",
    ylabel="Averaege Relative Distance",
    title="Average Relative Distance Per Generation",
)

fig_abs_distance, ax_abs_distance = plt.subplots()
ax_abs_distance.set(
    xlabel="Generation",
    ylabel="Averaege Absolute Distance",
    title="Average Absolute Distance Per Generation",
)

fig_collision, ax_collision = plt.subplots()
ax_collision.set(
    xlabel="Generation",
    ylabel="Averaege Collisions",
    title="Average Collision Per Generation",
)

for cross_rate in ls_cross_rate:
    GA = GeneticAlgorithm(
        n_robots=2 * 15, n_iter=epochs, cross_rate=cross_rate, mut_rate=1 / 5
    )  # n_robots should be an equal number for crossover to work

    best, best_eval = GA.main(60000)

    ax_fitness.plot(
        x, GA.get_results("fitness"), label=f"Crossover rate = {cross_rate}"
    )
    ax_collision.plot(x, GA.get_results("coll"), label=f"Crossover rate = {cross_rate}")
    ax_abs_distance.plot(
        x, GA.get_results("tot_dist"), label=f"Crossover rate = {cross_rate}"
    )
    ax_rel_distance.plot(
        x, GA.get_results("rel_dist"), label=f"Crossover rate = {cross_rate}"
    )


ax_fitness.legend(loc=1)
ax_collision.legend(loc=1)
ax_abs_distance.legend(loc=1)
ax_rel_distance.legend(loc=1)

plt.show()
