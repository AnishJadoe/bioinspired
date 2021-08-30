# Loading in functions
from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("bmh")

epochs = 10
x = range(0, epochs )
ls_cross_rate = np.linspace(0, 1, 6)
ls_mut_rate = 1 / np.linspace(1, 100, 10)

fig_fitness, ax_fitness = plt.subplots()
ax_fitness.set(xlabel="Generation",
               ylabel="Fitness",
               title="Average Fitness Per Generation")

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

for mut_rate in ls_mut_rate:
    print(f'USING A MUTATION RATE OF: {mut_rate}')
    GA = GeneticAlgorithm(
        n_robots=2 * 15, n_iter=epochs, cross_rate=0.8, mut_rate=mut_rate
    )  # n_robots should be an equal number for crossover to work

    best, best_eval = GA.main(60000)

    ax_fitness.plot(x,
                    GA.get_results("fitness"),
                    label=f"Mutation rate = {mut_rate}")
    ax_collision.plot(x,
                      GA.get_results("coll"),
                      label=f"Mutation rate = {mut_rate}")
    ax_abs_distance.plot(x,
                         GA.get_results("tot_dist"),
                         label=f"Mutation rate = {mut_rate}")
    ax_rel_distance.plot(x,
                         GA.get_results("rel_dist"),
                         label=f"Mutation rate = {mut_rate}")

ax_fitness.legend()
ax_collision.legend()
ax_abs_distance.legend()
ax_rel_distance.legend()

plt.show()
