import os
from ..world_map.world_map import WorldMap
from ..utility.functions import cache_size_kb, get_init_chromosomes_NN
from ..utility.run_simulation import run_simulation
from ..utility.constants import *
import math
import random
import numpy as np
import pickle


class GeneticAlgorithmRunner:
    """This class does all the work that is needed for a genetic algorithm,
    there is a mutation, crossover and selection operator available for which all the
    parameters can be changed as needed
    """

    def __init__(self, n_robots, epochs, run_time, cross_rate, mut_rate, seed):
        """Initialization of the class

        Args:
            n_robots (int): The amount of robots in the population
            epochs (int): The amount of epochs that the simulation will run for
            cross_rate (float): The crossover rate of the crossover operator
            mut_rate (float): The mutation rate of the mutation operator
        """
        self.n_robots = n_robots
        self.epochs = epochs
        self.fitness = list()
        self.best_eval = 0
        self.pop = get_init_chromosomes_NN(self.n_robots, N_INPUTS,N_OUTPUTS,N_HIDDEN)
        self.best_agent = self.pop[0]
        self.gen = 1
        self.run_time = run_time
        self.seed = seed
        
        self.best_fitness = list()
        self.best_individuals = list()

        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.clone = False
        self.converged = False
        self.reward_gen = list()
        self.coll_gen = list()
        self.dist_gen = list()
        self.rel_dist_gen = list()
        self.flips_gen = list()
        self.token_gen = list()
        self.results = dict()
        self.all_populations = dict()

        self.best_fitness_set= set()
        self.best_individual_set= set()

    def best_fit_selection(self):  # k=5
        """The selection operator used for this algorithm, it takes the best 20%
        of the generation and seperates it from the bottom 80%

        Returns:
            best_pop (list): The top 20% of the population
            other_pop (list): The bottom 80% of the population
        """

        index = min(-1, -math.floor(self.n_robots * 0.20))  # take best 20%
        best_index = np.argsort(np.array(self.fitness))[index:]
        best_pop = [self.pop[i] for i in best_index]
        other_index = np.argsort(np.array(self.fitness))[:index]

        other_pop = [self.pop[i] for i in other_index]

        return best_pop, other_pop
    
    def tournament_selection(self, k=3):
        selected = []
        for _ in range(self.n_robots):
            participants = random.sample(range(0,len(self.pop)-1),k)
            fitnesses = [self.fitness[p] for p in participants]
            selected.append(self.pop[participants[np.argmax(fitnesses)]])

        return selected

    def roulette_selection(self):
        ls_p = []
        if sum(self.fitness) == 0:
            self.converged = True
            return ls_p,ls_p
        # Create "roullete wheel"
        probabilities_wheel = self.fitness/sum(self.fitness)
        cum_sum = [0]
        for probability in probabilities_wheel:
            cum_sum.append(cum_sum[-1]+probability)
        cum_sum = np.array(cum_sum[1:])
        # Select parents
        for i in range(int(self.n_robots)):
            spin = np.random.uniform(0, 1)
            index_p = np.where(cum_sum>=spin)[0][0]
            ls_p.append(self.pop[index_p])
          

        return ls_p + ls_p



    def mutation(self, individual):
        """This is the mutation operator of the genetic algorithm, it takes an
        indivual and swaps 2 loci on the chromosome matrix

        Args:
            individual (numpy array): The individual on which the mutation will be preformed
        """
        random_choice = np.random.sample()
        if self.mut_rate > random_choice:
            # Two different types of mutation
            mutation_type = np.random.sample()
            if mutation_type >= 0.0 and mutation_type < 0.3:
                # Gene Swap
                idx1, idx2 = np.random.randint(0, individual.size, 2)
                individual.flat[idx1], individual.flat[idx2] = individual.flat[idx2], individual.flat[idx1]
            elif mutation_type >= 0.3 and mutation_type < 0.5:
                # Gene Inversion
                idx1, idx2 = np.random.randint(0, individual.size, 2)
                if idx1 < idx2:
                    inverted = reversed(individual.flat[idx1:idx2])
                    individual.flat[idx1:idx2] = list(inverted)
                else:
                    inverted = reversed(individual.flat[idx2:idx1])
                    individual.flat[idx2:idx1] = list(inverted)

            elif mutation_type >= 0.5 and mutation_type < 0.7:
                # Scramble Mutation
                idx1, idx2 = np.random.randint(0, individual.size, 2)
                if idx1 < idx2:
                    scramble = individual.flat[idx1:idx2]
                    random.shuffle(scramble)
                    individual.flat[idx1:idx2] = scramble
                else:
                    scramble = individual.flat[idx2:idx1]
                    random.shuffle(scramble)
                    individual.flat[idx2:idx1] = scramble
            else:
                # Random Resetting
                idx1 = np.random.randint(0,individual.size,1)
                weight_range = WEIGHT_RANGE
                individual.flat[idx1] = np.random.uniform(low=-weight_range, high=weight_range,size=1)
        return individual


    def two_point_crossover(self, parent1, parent2):
        size_chromosome = len(parent1.flatten())
        random_choice = np.random.sample()
        roll_1 = np.random.randint(low=0, high=size_chromosome)
        roll_2 = np.random.randint(low=0, high=size_chromosome)

        if roll_2 > roll_1:
            cross_point_1 = roll_1
            cross_point_2 = roll_2
        elif roll_1 > roll_2:
            cross_point_1 = roll_2
            cross_point_2 = roll_1
        else:
            cross_point_1 = roll_1
            cross_point_2 = roll_2 + 1

        child1 = parent1.copy().flatten()
        child2 = parent2.copy().flatten()
        self.clone = True

        if self.cross_rate > random_choice:
            self.clone = False
            snip_parent_1 = parent1.flatten()[cross_point_1:cross_point_2]
            snip_parent_2 = parent2.flatten()[cross_point_1:cross_point_2]

            child1[cross_point_1:cross_point_2] = snip_parent_2
            child2[cross_point_1:cross_point_2] = snip_parent_1

        else:
            child1, child2 = parent1, parent2

        return [child1.reshape(parent1.shape), child2.reshape(parent2.shape)]
        
    def get_best_individuals(self):

        # Number of elite individuals to select (top 10%)
        num_elite = max(1, math.ceil(self.n_robots * 0.10))

        # Get indices of sorted fitness values in descending order
        sorted_indices = np.argsort(self.fitness)[::-1]

        # Select the top 10% individuals
        elitist_children = sorted_indices[:num_elite]

        for index in elitist_children:
            fitness_value = self.fitness[index]
            individual_str = str(self.pop[index].tolist())  # Convert individual to string for comparison
            
            if fitness_value not in self.best_fitness_set and individual_str not in self.best_individual_set:
                if len(self.best_fitness) < num_elite:
                    # Add to best lists if they are not full
                    self.best_fitness.append(fitness_value)
                    self.best_individuals.append(self.pop[index])
                    self.best_fitness_set.add(fitness_value)
                    self.best_individual_set.add(individual_str)
                else:
                    # Replace the worst in the best lists if current individual is better
                    min_fitness_index = np.argmin(self.best_fitness)
                    if fitness_value > self.best_fitness[min_fitness_index]:
                        removed_fitness = self.best_fitness[min_fitness_index]
                        removed_individual = str(self.best_individuals[min_fitness_index].tolist())
                        
                        self.best_fitness[min_fitness_index] = fitness_value
                        self.best_individuals[min_fitness_index] = self.pop[index]
                        
                        # Update sets
                        self.best_fitness_set.remove(removed_fitness)
                        self.best_individual_set.remove(removed_individual)
                        
                        self.best_fitness_set.add(fitness_value)
                        self.best_individual_set.add(individual_str)

        # Sort best_fitness and best_individuals together based on fitness values
        best_sorted = sorted(zip(self.best_fitness, self.best_individuals), key=lambda x: x[0], reverse=True)

        # Keep only the top num_elite individuals
        self.best_fitness, self.best_individuals = zip(*best_sorted[:num_elite])
        self.best_fitness = list(self.best_fitness)
        self.best_individuals = list(self.best_individuals)

        print(f"Current best individuals have fitness {self.best_fitness}")
        return self.best_individuals

    def run(self):
        """This is the main loop for the algorithm. First a base score is set-up by running the algorithm with the inital population

        Args:
            sim_time ([type]): [description]

        Returns:
            [type]: [description]
        """
        for epoch in range(self.epochs):
            print(f"GENERATION: {self.gen}")
            self._save_population()
            # Build world
            wm = WorldMap(skeleton_file="bioinspired/src/world_map/maps/follow_token_map.txt", 
                          map_width=MAP_DIMS[0], map_height=MAP_DIMS[1], tile_size=CELL_SIZE)
            population_results = run_simulation(
                wm, self.run_time, self.pop, self.n_robots, self.gen
            )
    
            self._save_results(population_results)
            self.fitness = population_results["pop_fitness"]
            for i in range(self.n_robots):
                if self.fitness[i] > self.best_eval:
                    self.best_agent, self.best_eval = self.pop[i], self.fitness[i]
                    print(
                        f"Generation {self.gen} gives a new best \
                            with score {self.fitness[i]}"
                    )

            children = list()
            # elitism
            best_individuals = self.get_best_individuals()
            children.extend(best_individuals)
            parents = self.tournament_selection()

            mating = True
            while mating:
                parent1_index = np.random.randint(0, len(parents))
                parent2_index = np.random.randint(0, len(parents))
                parent1 = parents[parent1_index]
                parent2= parents[parent2_index]

                for c in self.two_point_crossover(parent1, parent2):
                    self.mutation(c)
                    children.append(c)

                if len(children) >= self.n_robots:
                    mating = False

            self.pop = children[:self.n_robots+1]
            self.gen += 1
            print(f"Size of cache is {cache_size_kb()/1000} MB")
            if self.gen % 50 == 0 and epoch != (self.epochs-1): # Save every 10 generations
                self.save_run()

        self.save_run()

        return

    def _save_population(self):
        self.all_populations[self.gen] = self.pop

    def _save_results(self, population_results):
        results_this_gen = {}

        results_this_gen["fitness"] = population_results["pop_fitness"]
        results_this_gen["abs_dist"] = population_results["pop_abs_distance"]
        results_this_gen["rel_dist"] = population_results["pop_rel_distance"]
        results_this_gen["collisions"] = population_results["pop_collisions"]
        results_this_gen["tokens"] = population_results["pop_token"]
        results_this_gen["cells_explored"] = population_results["pop_cells_explored"]
        results_this_gen["best_eval"] = self.best_eval
        results_this_gen["best_agent"] = self.best_agent

        self.results[self.gen] = results_this_gen

    def save_run(self):
        self.world_map = []  # Can't save pygame surface
        folder = f"saved_runs/n_robots_{self.n_robots}/epochs_{self.gen-1}/epoch_time_{self.run_time}/seed_{self.seed}/neurons_{N_HIDDEN}"
        run_name = f"{self.mut_rate}mut_{self.cross_rate}cross.pkl"
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/{run_name}", "wb") as f:
            pickle.dump(self, f)

    def best_gen(self):
        return self.best_agent
