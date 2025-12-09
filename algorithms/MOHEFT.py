from config import *
import random

class MOHEFT:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data
        self.num_tasks = self.data['User'].count()
        self.num_resources = self.data['EdgeServer'].count()
        self.gene_size = self.num_tasks * self.num_resources

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = Individual()
            individual.CInd = []
            # Correct One-Hot Initialization
            for _ in range(self.num_tasks):
                task_gene = [0] * self.num_resources
                chosen_server = random.randint(0, self.num_resources - 1)
                task_gene[chosen_server] = 1
                individual.CInd.extend(task_gene)
            population.append(individual)
        return population

    def tournament_selection(self, population):
        a, b = random.sample(population, 2)
        if a.rank < b.rank: return a
        elif b.rank < a.rank: return b
        return a if a.crowding_distance > b.crowding_distance else b

    def crossover(self, p1, p2):
        c1, c2 = Individual(), Individual()
        pt = random.randint(1, self.gene_size - 2)
        c1.CInd = p1.CInd[:pt] + p2.CInd[pt:]
        c2.CInd = p2.CInd[:pt] + p1.CInd[pt:]
        return c1, c2

    def mutation(self, ind):
        if random.random() < 0.1:
            idx = random.randint(0, self.gene_size - 1)
            ind.CInd[idx] = 1 - ind.CInd[idx]
        return ind

    def non_dominated_sorting(self, population):
        for p in population:
            p.domination_count = 0
            p.dominated_set = []
            for q in population:
                if all(x <= y for x, y in zip(p.fitness, q.fitness)) and any(x < y for x, y in zip(p.fitness, q.fitness)):
                    p.dominated_set.append(q)
                elif all(y <= x for x, y in zip(p.fitness, q.fitness)) and any(y < x for x, y in zip(p.fitness, q.fitness)):
                    p.domination_count += 1
            if p.domination_count == 0: p.rank = 0
        return [p for p in population if p.rank == 0]

    def select_best_population(self, combined_population):
        combined_population.sort(key=lambda x: sum(x.fitness))
        return combined_population[:self.population_size]

    def run(self):
        population = self.initialize_population()
        population = self.fitness(population, self.data)
        for _ in range(self.generation_count):
            offspring = []
            while len(offspring) < self.population_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                c1, c2 = self.crossover(p1, p2)
                offspring.extend([self.mutation(c1), self.mutation(c2)])
            offspring = self.fitness(offspring, self.data)
            combined = population + offspring
            population = self.select_best_population(combined)
        return population