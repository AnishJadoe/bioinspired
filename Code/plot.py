# Loading in functions
import matplotlib.pyplot as plt
import numpy as np
import math
from genetic_algorithm import GeneticAlgorithmRunner

plt.style.use("bmh")


def split_array(array):
    # Sort the array in descending order
    sorted_array = sorted(array, reverse=True)

    # Calculate the index to split the array into 20% largest and 80% smallest
    split_index = int(len(sorted_array) * 0.2)

    # Split the array into two parts
    largest_20_percent = np.mean(sorted_array[:split_index])
    smallest_80_percent = np.mean(sorted_array[split_index:])

    return largest_20_percent, smallest_80_percent


def plot_mean_results(GA: GeneticAlgorithmRunner):
    x = range(0, GA.epochs)
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

    fig_token, ax_token = plt.subplots()
    ax_token.set(
        xlabel="Generation",
        ylabel="Averaege Tokens",
        title="Average Token Per Generation",
    )

    fitness = [np.mean(GA.results[key]["fitness"]) for key in GA.results]
    collisions = [np.mean(GA.results[key]["collisions"]) for key in GA.results]
    abs_distance = [np.mean(GA.results[key]["abs_dist"]) for key in GA.results]
    rel_distance = [np.mean(GA.results[key]["rel_dist"]) for key in GA.results]
    tokens = [np.mean(GA.results[key]["tokens"]) for key in GA.results]

    ax_fitness.plot(x, fitness, label=f"Mutation rate = {GA.mut_rate}")
    ax_collision.plot(x, collisions, label=f"Mutation rate = {GA.mut_rate}")
    ax_abs_distance.plot(x, abs_distance, label=f"Mutation rate = {GA.mut_rate}")
    ax_rel_distance.plot(x, rel_distance, label=f"Mutation rate = {GA.mut_rate}")
    ax_token.plot(x, tokens, label=f"Mutation rate = {GA.mut_rate}")

    ax_fitness.legend()
    ax_collision.legend()
    ax_abs_distance.legend()
    ax_rel_distance.legend()
    ax_token.legend()
    plt.show()


def plot_top_20_results(GA: GeneticAlgorithmRunner):
    x = range(0, GA.epochs)
    fig_fitness, ax_fitness = plt.subplots()
    ax_fitness.set(
        xlabel="Generation", ylabel="Fitness", title="Top vs Bot Fitness Per Generation"
    )

    fig_rel_distance, ax_rel_distance = plt.subplots()
    ax_rel_distance.set(
        xlabel="Generation",
        ylabel="Averaege Relative Distance",
        title="Top vs Bot Per Generation",
    )

    fig_abs_distance, ax_abs_distance = plt.subplots()
    ax_abs_distance.set(
        xlabel="Generation",
        ylabel="Averaege Absolute Distance",
        title="Top vs Bot Per Generation",
    )

    fig_collision, ax_collision = plt.subplots()
    ax_collision.set(
        xlabel="Generation",
        ylabel="Averaege Collisions",
        title="Top vs Bot Generation",
    )

    fig_token, ax_token = plt.subplots()
    ax_token.set(
        xlabel="Generation",
        ylabel="Averaege Tokens",
        title="Top vs Bot Generation",
    )
    
    fitness_top = [split_array(GA.results[key]["fitness"])[0] for key in GA.results]
    collisions_top = [split_array(GA.results[key]["collisions"])[0] for key in GA.results]
    abs_distance_top = [split_array(GA.results[key]["abs_dist"])[0] for key in GA.results]
    rel_distance_top = [split_array(GA.results[key]["rel_dist"])[0] for key in GA.results]
    tokens_top = [split_array(GA.results[key]["tokens"])[0] for key in GA.results]
    
    
    fitness_bot = [split_array(GA.results[key]["fitness"])[0] for key in GA.results]
    collisions_bot = [split_array(GA.results[key]["collisions"])[0] for key in GA.results]
    abs_distance_bot = [split_array(GA.results[key]["abs_dist"])[0] for key in GA.results]
    rel_distance_bot = [split_array(GA.results[key]["rel_dist"])[0] for key in GA.results]
    tokens_bot = [split_array(GA.results[key]["tokens"])[0] for key in GA.results]

    ax_fitness.plot(x, fitness_top, label="Top percentage of the population")
    ax_collision.plot(x, collisions_top, label="Top percentage of the population")
    ax_abs_distance.plot(x, abs_distance_top, label="Top percentage of the population")
    ax_rel_distance.plot(x, rel_distance_top, label="Top percentage of the population")
    ax_token.plot(x, tokens_top, label="Top percentage of the population")
    
    ax_fitness.plot(x, fitness_bot, label="Bot percentage of the population")
    ax_collision.plot(x, collisions_bot, label="Bot percentage of the population")
    ax_abs_distance.plot(x, abs_distance_bot, label="Bot percentage of the population")
    ax_rel_distance.plot(x, rel_distance_bot, label="Bot percentage of the population")
    ax_token.plot(x, tokens_bot, label="Bot percentage of the population")

    ax_fitness.legend()
    ax_collision.legend()
    ax_abs_distance.legend()
    ax_rel_distance.legend()
    ax_token.legend()
    plt.show()
