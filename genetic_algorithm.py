from functions import get_init_pop
from run_simulation import run_simulation


class GeneticAlgorithm:
    
    def __init__(self, n_robots, n_iter, cross_rate, mut_rate):

        self.n_robots = n_robots
        self.n_iter = n_iter
        self.scores = list()
        self.best_eval = 0
        self.pop = get_init_pop(self.n_robots)
        self.best = self.pop[0]

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
        '''
        Using a tournement style method, we obtain the best
        agent in that population.
        :param pop:
        :param scores:
        :param k:
        :return:
        '''

        index = min(-1, -math.floor(self.n_robots * 0.20))  # take best 10%
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
        random_choice = np.random.sample()

        if random_choice > self.mut_rate:

            unequal = True

            while unequal:
                # Row and column to start the mutation swap with
                random_row_s = np.random.randint(low=0, high=individual.shape[0])
                random_column_s = np.random.randint(low=0, high=individual.shape[1])

                # Row and column to be swapped
                random_row_e = np.random.randint(low=0, high=individual.shape[0])
                random_column_e = np.random.randint(low=0, high=individual.shape[1])

                if (random_row_s != random_row_e) or (random_column_s != random_column_e):
                    unequal = False

            p_start = individual[random_row_s, random_column_s]
            p_end = individual[random_row_e, random_column_e]

            individual[random_row_s, random_column_s] = p_end
            individual[random_row_e, random_column_e] = p_start

        return

    def crossover(self, parent1, parent2):

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

        # Construct initial population, make sure to figure out how to make this
        # flow more logically

        print('GENERATION 0')

        self.scores = run_simulation(sim_time, self.pop, self.n_robots, self)
        self.best_eval = self.scores[0]

        for gen in range(self.n_iter):
            self.gen += 1
            print(f'GENERATION: {gen + 1}')
            self.scores = run_simulation(sim_time, self.pop, self.n_robots, self)

            for i in range(self.n_robots):
                if self.scores[i] > self.best_eval:
                    self.best, self.best_eval = self.pop[i], self.scores[i]
                    print(f'Generation {gen + 1} gives a new best with score {self.scores[i]}')
            # selected = [self.selection() for _ in range(self.n_robots)]
            ls_p1, ls_p2 = self.selection()

            children = list()
            mating = True
            while mating:

                random_parent1 = np.random.randint(0, len(ls_p1))
                parent1 = ls_p1[random_parent1]
                flip = np.random.uniform(0, 1)

                if flip > 0.4:
                    random_parent2 = np.random.randint(0, len(ls_p1))
                    parent2 = ls_p1[random_parent2]
                else:
                    random_parent2 = np.random.randint(0, len(ls_p2))
                    parent2 = ls_p2[random_parent2]

                for c in self.crossover(parent1, parent2):

                    if not self.clone:
                        self.mutation(c)
                        children.append(c)
                    else:
                        children.append(c)

                if len(children) >= self.n_robots:
                    mating = False

            # for i in range(0, self.n_robots, 2):
            #     p1, p2 = selected[i], selected[i + 1]
            #
            #     for c in self.crossover(p1, p2):
            #         if not self.clone:
            #             self.mutation(c)
            #             children.append(c)
            #         else:
            #             children.append(c)
            self.pop = children

        return self.best, self.best_eval

    def get_results(self, result):

        if result == 'fitness':
            result = self.reward_gen

        if result == 'tot_dist':
            result = self.dist_gen

        if result == 'rel_dist':
            result = self.rel_dist_gen

        if result == 'flip':
            result = self.flips_gen

        if result == 'coll':
            result = self.coll_gen

        if result == 'token':
            result = self.token_gen

        return result

    def best_gen(self):
        return self.best