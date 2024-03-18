from genetic_algorithm import GeneticAlgorithm


ga = GeneticAlgorithm(n_robots=5,n_iter=20,cross_rate=0.5,mut_rate=0.1)
ga.main(30000)
