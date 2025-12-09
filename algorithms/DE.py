from config import *
import random
import numpy as np
import copy

class DE:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data
        self.num_tasks = self.data['User'].count()
        self.num_resources = self.data['EdgeServer'].count()
        self.gene_len = self.num_tasks * self.num_resources
        
        self.F = 0.5  # Mutation factor
        self.CR = 0.7 # Crossover probability

    def get_score(self, individual):
        if not individual.fitness or individual.fitness[0] == float('inf'): return float('inf')
        return sum(individual.fitness)

    def continuous_to_discrete(self, vector):
        c_ind = []
        for i in range(self.num_tasks):
            start = i * self.num_resources
            end = start + self.num_resources
            segment = vector[start:end]
            chosen = np.argmax(segment)
            gene = [0] * self.num_resources
            gene[chosen] = 1
            c_ind.extend(gene)
        return c_ind

    def discrete_to_continuous(self, individual):
        return np.array(individual.CInd, dtype=float)

    def run(self):
        # 1. Initialize Vectors
        vectors = []
        population = []
        
        for _ in range(self.population_size):
            # Random continuous vectors [0, 1]
            vec = np.random.rand(self.gene_len)
            vectors.append(vec)
            
            ind = Individual()
            ind.CInd = self.continuous_to_discrete(vec)
            population.append(ind)
            
        population = self.fitness(population, self.data)

        # 2. Evolution
        for _ in range(self.generation_count):
            new_population = []
            new_vectors = []
            
            for i in range(self.population_size):
                # Mutation: Select 3 random distinct vectors
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                
                target_vec = vectors[i]
                donor_vec = vectors[a] + self.F * (vectors[b] - vectors[c])
                
                # Crossover (Binomial)
                trial_vec = np.zeros_like(target_vec)
                for j in range(self.gene_len):
                    if random.random() <= self.CR or j == random.randint(0, self.gene_len - 1):
                        trial_vec[j] = donor_vec[j]
                    else:
                        trial_vec[j] = target_vec[j]
                
                # Create Trial Individual
                trial_ind = Individual()
                trial_ind.CInd = self.continuous_to_discrete(trial_vec)
                
                # Selection (Greedy)
                # We need to evaluate the trial individual immediately
                # Note: This is computationally expensive, so we usually evaluate the whole batch.
                # For simplicity in this architecture, we will add to a temporary list.
                new_population.append(trial_ind)
                new_vectors.append(trial_vec) # Potential vector

            # Batch Evaluate
            new_population = self.fitness(new_population, self.data)
            
            # Selection Step
            final_population = []
            final_vectors = []
            
            for i in range(self.population_size):
                score_old = self.get_score(population[i])
                score_new = self.get_score(new_population[i])
                
                if score_new < score_old:
                    final_population.append(new_population[i])
                    final_vectors.append(new_vectors[i])
                else:
                    final_population.append(population[i])
                    final_vectors.append(vectors[i])
            
            population = final_population
            vectors = final_vectors

        return population