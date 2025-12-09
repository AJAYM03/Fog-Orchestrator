from config import *
import random
import copy
import numpy as np

class GA:
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
            for _ in range(self.num_tasks):
                task_gene = [0] * self.num_resources
                chosen_server = random.randint(0, self.num_resources - 1)
                task_gene[chosen_server] = 1
                individual.CInd.extend(task_gene)
            population.append(individual)
        return population

    def get_score(self, individual):
        # Scalarize the 3 objectives (Equal weight)
        # We use the normalized values from individual.fitness
        if not individual.fitness or individual.fitness[0] == float('inf'):
            return float('inf')
        return sum(individual.fitness)

    def selection(self, population):
        # Tournament Selection
        competitors = random.sample(population, 3)
        competitors.sort(key=self.get_score)
        return competitors[0]

    def crossover(self, p1, p2):
        # Uniform Crossover
        c1, c2 = Individual(), Individual()
        mask = [random.randint(0, 1) for _ in range(self.gene_size)]
        c1_genes, c2_genes = [], []
        
        for i in range(self.gene_size):
            if mask[i] == 0:
                c1_genes.append(p1.CInd[i])
                c2_genes.append(p2.CInd[i])
            else:
                c1_genes.append(p2.CInd[i])
                c2_genes.append(p1.CInd[i])
        
        c1.CInd = c1_genes
        c2.CInd = c2_genes
        return c1, c2

    def mutation(self, individual):
        mutation_rate = 1.0 / self.gene_size
        new_genes = individual.CInd[:]
        for i in range(self.gene_size):
            if random.random() < mutation_rate:
                new_genes[i] = 1 - new_genes[i]
        individual.CInd = new_genes
        return individual

    def run(self):
        population = self.initialize_population()
        population = self.fitness(population, self.data)
        
        for _ in range(self.generation_count):
            offspring = []
            while len(offspring) < self.population_size:
                p1 = self.selection(population)
                p2 = self.selection(population)
                c1, c2 = self.crossover(p1, p2)
                offspring.extend([self.mutation(c1), self.mutation(c2)])
            
            offspring = self.fitness(offspring, self.data)
            
            # Elitism: Combine and sort by scalar score
            combined = population + offspring
            combined.sort(key=self.get_score)
            population = combined[:self.population_size]
            
        return population