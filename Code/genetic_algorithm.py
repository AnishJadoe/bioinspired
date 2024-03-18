from functions import get_init_pop
from run_simulation import run_simulation
import math
import numpy as np


class GeneticAlgorithm:
    """This class does all the work that is needed for a genetic algorithm,
    there is a mutation, crossover and selection operator available for which all the 
    parameters can be changed as needed 
    """

    def __init__(self, n_robots, n_iter, cross_rate, mut_rate):
        """Initialization of the class 

        Args:
            n_robots (int): The amount of robots in the population
            n_iter (int): The amount of epochs that the simulation will run for
            cross_rate (float): The crossover rate of the crossover operator 
            mut_rate (float): The mutation rate of the mutation operator 
        """

        self.n_robots = n_robots
        self.n_iter = n_iter
        self.scores = list()
        self.best_eval = 0
        self.pop = get_init_pop(self.n_robots)
        self.best = self.pop[0]
        self.best_results = {}
        self.gen = 0

        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.clone = False

        self.reward_gen = list()
        self.coll_gen = list()
        self.dist_gen = list()
        self.rel_dist_gen = list()
        self.flips_gen = list()
        self.token_gen = list()

    def selection(self):  # k=5
        """The selection operator used for this algorithm, it takes the best 20%
        of the generation and seperates it from the bottom 80%

        Returns:
            best_pop (list): The top 20% of the population
            other_pop (list): The bottom 80% of the population 
        """

        index = min(-1, -math.floor(self.n_robots * 0.20))  # take best 20%
        best_index = np.argsort(np.array(self.scores))[index:]
        best_pop = [self.pop[i] for i in best_index]
        other_index = np.argsort(np.array(self.scores))[:index]

        other_pop = [self.pop[i] for i in other_index]

        # selection_ix = np.random.randint(len(self.pop))
        # for ix in np.random.randint(0, len(self.pop), k - 1):
        #     if self.ls_rewards[ix] < self.ls_rewards[selection_ix]:
        #         selection_ix = ix

        return best_pop, other_pop

    def mutation(self, individual):
        """This is the mutation operator of the genetic algorithm, it takes an
        indivual and swaps 2 loci on the chromosome matrix 

        Args:
            individual (numpy array): The individual on which the mutation will be preformed 
        """
        random_choice = np.random.sample()

        if self.mut_rate > random_choice:

            unequal = True

            while unequal:
                # Row and column to start the mutation swap with
                random_row_s = np.random.randint(low=0, high=individual.shape[0])
                random_column_s = np.random.randint(low=0, high=individual.shape[1])

                # Row and column with which we want to swap the orginal data point with 
                random_row_e = np.random.randint(low=0, high=individual.shape[0])
                random_column_e = np.random.randint(low=0, high=individual.shape[1])

                if (random_row_s != random_row_e) or (
                    random_column_s != random_column_e
                ):
                    # only if the rows or column are not the same we will swap the data points 
                    unequal = False 

            p_start = individual[random_row_s, random_column_s]
            p_end = individual[random_row_e, random_column_e]

            individual[random_row_s, random_column_s] = p_end
            individual[random_row_e, random_column_e] = p_start

        return

    def two_point_crossover(self,parent1,parent2):
        size_chromosome = len(parent1.flatten())
        random_choice = np.random.sample()
        cross_point_1 = np.random.randint(low=0,high=size_chromosome)
        cross_point_2 = np.random.randint(low=cross_point_1,high=size_chromosome)

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
        
    def crossover(self, parent1, parent2):
        """The crossover operator of the algorithm, it takes 2 parent chromosomes and 
        uses either a horizontal (row) or vertical (column) method  for crossover 

        Args:
            parent1 (numpy array): The first parent used for crossover 
            parent2 (numpy array): The second parent used for crossover 

        Returns:
            [child1, child2] (list): A list containing both the first and second child
            obtained after crossover 
        """

        random_choice = np.random.sample()
        random_row = np.random.randint(low=1, high=parent1.shape[0])
        random_column = np.random.randint(low=1, high=parent1.shape[1])

        empty_child1 = np.zeros(parent1.shape)
        empty_child2 = np.zeros(parent1.shape)
        self.clone = True

        if self.cross_rate > random_choice:
            self.clone = False

            # Horizontal crossover
            if random_choice > 0.5:
                empty_child1[:random_row, :] = parent1[:random_row, :]
                empty_child1[random_row:, :] = parent2[random_row:, :]
                child1 = empty_child1

                empty_child2[:random_row, :] = parent2[:random_row, :]
                empty_child2[random_row:, :] = parent1[random_row:, :]
                child2 = empty_child2
            # Vertical crossover
            else:
                empty_child1[:, :random_column] = parent1[:, :random_column]
                empty_child1[:, random_column:] = parent2[:, random_column:]
                child1 = empty_child1

                empty_child2[:, :random_column] = parent2[:, :random_column]
                empty_child2[:, random_column:] = parent1[:, random_column:]
                child2 = empty_child2
        else:
            child1, child2 = parent1, parent2

        return [child1, child2]

    def main(self, sim_time):
        """This is the main loop for the algorithm. First a base score is set-up by running the algorithm with the inital population

        Args:
            sim_time ([type]): [description]

        Returns:
            [type]: [description]
        """

        for gen in range(self.n_iter):
            
            print(f"GENERATION: {gen}")
            self.scores = run_simulation(sim_time, self.pop, self.n_robots, self)
            
            for i in range(self.n_robots):
                if self.scores[i] > self.best_eval:
                    self.best, self.best_eval = self.pop[i], self.scores[i]
                    print(
                        f"Generation {gen} gives a new best with score {self.scores[i]}"
                    )
            
            # selected = [self.selection() for _ in range(self.n_robots)]
            ls_p1, ls_p2 = self.selection()

            children = list()
            mating = True
            while mating:

                random_parent1 = np.random.randint(0, len(ls_p1))
                parent1 = ls_p1[random_parent1]
                flip = np.random.uniform(0, 1)

                if flip > 0.6:
                    # Let the parent mate with another parent from the top 20%

                    random_parent2 = np.random.randint(0, len(ls_p1))
                    parent2 = ls_p1[random_parent2]
                else:
                    # Let the parent mate with a parent from the bottom 80%

                    random_parent2 = np.random.randint(0, len(ls_p2))
                    parent2 = ls_p2[random_parent2]

                for c in self.two_point_crossover(parent1, parent2):
                    self.mutation(c)
                    children.append(c)


                if len(children) >= self.n_robots:
                    mating = False

            self.best_results[f'generation_{gen}'] = (self.best,self.best_eval)
            self.pop = children
            self.gen += 1

        return self.best, self.best_eval

    def get_results(self, result):

        if result == "fitness":
            result = self.reward_gen

        if result == "tot_dist":
            result = self.dist_gen

        if result == "rel_dist":
            result = self.rel_dist_gen

        if result == "flip":
            result = self.flips_gen

        if result == "coll":
            result = self.coll_gen

        if result == "token":
            result = self.token_gen

        return result

    def best_gen(self):
        return self.best