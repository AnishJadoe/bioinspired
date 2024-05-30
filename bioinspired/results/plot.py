# Loading in functions
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from src.runners.genetic_algorithm import GeneticAlgorithmRunner

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

def plot_diversity(GA:GeneticAlgorithmRunner):
    fig,ax_fitness = plt.subplots()
    ax_fitness.set(
        xlabel="Fitness",
        ylabel="Probability",
        title="Distribution of fitness per generation"
    )
    fig,ax_distance = plt.subplots()
    ax_distance.set(
        xlabel="Distance Travelled",
        ylabel="Probability",
        title="Distribution of distance travelled per generation"
    )

    fig,ax_collisions = plt.subplots()
    ax_collisions.set(
        xlabel="Collisions",
        ylabel="Probability",
        title="Distribution of collisions per generation"
    )
    
    for generation, result in GA.results.items():
        ax_fitness.hist(x=result["fitness"],label=f'Generation {generation}',alpha=0.7)
        #ax_fitness.vlines(x=np.mean(result["fitness"]), ymin=0,ymax=100)

        ax_distance.hist(x=result["abs_dist"],label=f'Generation {generation}',alpha=0.7)
        #ax_distance.vlines(x=np.mean(result["abs_dist"]), ymin=0,ymax=100)

        ax_collisions.hist(x=result["collisions"],label=f'Generation {generation}',alpha=0.7)
        #ax_collisions.vlines(x=np.mean(result["collisions"]), ymin=0,ymax=100)

    ax_fitness.legend() 
    ax_distance.legend()
    ax_collisions.legend()
    
    plt.show()

def plot_mean_results(GAs: GeneticAlgorithmRunner):
    fig_fitness, ax_fitness = plt.subplots()
    ax_fitness.set(
        xlabel="Generation", ylabel="Fitness", title="Average Fitness Per Generation"
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
    if type(GAs) != list:
        GAs = [GAs]
    for GA in GAs:
        x = range(0, GA.epochs)
        fitness = [np.mean(GA.results[key]["fitness"]) for key in GA.results]
        collisions = [np.mean(GA.results[key]["collisions"]) for key in GA.results]
        abs_distance = [np.mean(GA.results[key]["abs_dist"]) for key in GA.results]
        tokens = [np.mean(GA.results[key]["tokens"]) for key in GA.results]

        ax_fitness.plot(x, fitness, label=f"Mutation rate = {GA.mut_rate}")
        ax_collision.plot(x, collisions, label=f"Mutation rate = {GA.mut_rate}")
        ax_abs_distance.plot(x, abs_distance, label=f"Mutation rate = {GA.mut_rate}")
        ax_token.plot(x, tokens, label=f"Mutation rate = {GA.mut_rate}")

    ax_fitness.legend()
    ax_collision.legend()
    ax_abs_distance.legend()
    ax_token.legend()
    plt.show()

def plot_aggregate_results(GAs: List[GeneticAlgorithmRunner]):
    fig_fitness, ax_fitness = plt.subplots()
    ax_fitness.set(
        xlabel="Generation", ylabel="Fitness", title="Average Fitness Per Generation"
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
    if type(GAs) != list:
        GAs = [GAs]
    
    aggregate_fitness = {} # Data structure to agglomorate the results
    aggregate_collisions = {} # Data structure to agglomorate the results
    aggregate_abs_distance = {} # Data structure to agglomorate the results
    aggregate_tokens = {} # Data structure to agglomorate the results
    for GA in GAs:
        x = range(0, GA.epochs)
        if not GA.mut_rate in aggregate_fitness:
            aggregate_fitness[GA.mut_rate] = np.zeros((1,30))
            aggregate_collisions[GA.mut_rate] = np.zeros((1,30))
            aggregate_abs_distance[GA.mut_rate] = np.zeros((1,30))
            aggregate_tokens[GA.mut_rate] = np.zeros((1,30))
            
        fitness = [np.mean(GA.results[key]["fitness"]) for key in GA.results]
        collisions = [np.mean(GA.results[key]["collisions"]) for key in GA.results]
        abs_distance = [np.mean(GA.results[key]["abs_dist"]) for key in GA.results]
        tokens = [np.mean(GA.results[key]["tokens"]) for key in GA.results]

        aggregate_fitness[GA.mut_rate] = np.vstack((aggregate_fitness[GA.mut_rate],fitness))
        aggregate_collisions[GA.mut_rate] = np.vstack((aggregate_collisions[GA.mut_rate],collisions))
        aggregate_abs_distance[GA.mut_rate] = np.vstack((aggregate_abs_distance[GA.mut_rate],abs_distance))
        aggregate_tokens[GA.mut_rate] = np.vstack((aggregate_tokens[GA.mut_rate],tokens))


    for mut_rate in aggregate_fitness:
        fitness = aggregate_fitness[mut_rate][1:,:].mean(axis=0)
        fitness_std = aggregate_fitness[mut_rate][1:,:].std(axis=0)

        collisions = aggregate_collisions[mut_rate][1:,:].mean(axis=0)
        collisions_std = aggregate_collisions[mut_rate][1:,:].std(axis=0)

        abs_distance = aggregate_abs_distance[mut_rate][1:,:].mean(axis=0)
        abs_distance_std = aggregate_abs_distance[mut_rate][1:,:].std(axis=0)

        tokens = aggregate_tokens[mut_rate][1:,:].mean(axis=0)
        tokens_std = aggregate_tokens[mut_rate][1:,:].std(axis=0)

        ax_fitness.plot(x, fitness, label=f"Mutation rate = {mut_rate}")
        ax_fitness.fill_between(x, fitness-fitness_std, fitness+fitness_std ,alpha=0.3)
        
        ax_collision.plot(x, collisions, label=f"Mutation rate = {mut_rate}")
        ax_collision.fill_between(x, collisions-collisions_std, collisions+collisions_std ,alpha=0.3)

        ax_abs_distance.plot(x, abs_distance, label=f"Mutation rate = {mut_rate}")
        ax_abs_distance.fill_between(x, abs_distance-abs_distance_std, abs_distance+abs_distance_std ,alpha=0.3)

        ax_token.plot(x, tokens, label=f"Mutation rate = {mut_rate}")
        ax_token.fill_between(x, tokens-tokens_std, tokens+tokens_std ,alpha=0.3)

    ax_fitness.legend()
    ax_collision.legend()
    ax_abs_distance.legend()
    ax_token.legend()
    plt.show()

def plot_best_results(GAs: List[GeneticAlgorithmRunner]):
    fig_fitness, ax_fitness = plt.subplots()
    ax_fitness.set(
        xlabel="Generation", ylabel="Fitness", title="Best Fitness Per Generation"
    )

    fig_abs_distance, ax_abs_distance = plt.subplots()
    ax_abs_distance.set(
        xlabel="Generation",
        ylabel="Absolute Distance",
        title="Best Absolute Distance Per Generation",
    )

    fig_collision, ax_collision = plt.subplots()
    ax_collision.set(
        xlabel="Generation",
        ylabel="Collisions",
        title="Best Collision Per Generation",
    )

    fig_token, ax_token = plt.subplots()
    ax_token.set(
        xlabel="Generation",
        ylabel="Tokens",
        title="Tokens Per Generation",
    )
    if type(GAs) != list:
        GAs = [GAs]

    for GA in GAs:
        x = range(0, GA.epochs)
        fitness = [np.max(GA.results[key]["fitness"]) for key in GA.results]
        collisions = [np.min(GA.results[key]["collisions"]) for key in GA.results]
        abs_distance = [np.max(GA.results[key]["abs_dist"]) for key in GA.results]
        tokens = [np.max(GA.results[key]["tokens"]) for key in GA.results]

        ax_fitness.plot(x, fitness, label=f"Mutation rate = {GA.mut_rate}")
        ax_collision.plot(x, collisions, label=f"Mutation rate = {GA.mut_rate}")
        ax_abs_distance.plot(x, abs_distance, label=f"Mutation rate = {GA.mut_rate}")
        ax_token.plot(x, tokens, label=f"Mutation rate = {GA.mut_rate}")

    ax_fitness.legend()
    ax_collision.legend()
    ax_abs_distance.legend()
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

def plot_motorspeed_hist(GA:GeneticAlgorithmRunner):
    fig,ax = plt.subplots()
    for population in GA.all_populations:
        data = np.array(GA.all_populations[population]).reshape(-1,32,2)
        mean_representation = data.mean(axis=0)
        right_motor_speed = mean_representation[:,0]
        left_motor_speed = mean_representation[:,1]
        ax.hist(left_motor_speed + right_motor_speed, label=f'Generation {population} ',alpha=0.7)
    ax.legend()
    plt.show()

def plot_motorspeed(GA:GeneticAlgorithmRunner):
    fig,ax = plt.subplots()
    for population in GA.all_populations:
        data = np.array(GA.all_populations[population]).reshape(-1,32,2)
        mean_representation = data.mean(axis=0)
        right_motor_speed = mean_representation[:,0]
        left_motor_speed = mean_representation[:,1]
        ax.plot(np.arange(0,32), left_motor_speed + right_motor_speed,label=f'Generation {population} rms',alpha=0.7)    
    ax.legend()
    plt.show()