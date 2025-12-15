import random
from config import *

class RA:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data
        
        self.all_tasks = get_all_tasks(data)
        self.num_tasks = len(self.all_tasks)

    def schedule(self):
        individual = Individual()
        num_servers = self.data['EdgeServer'].count()
        individual.CInd = [0] * (self.num_tasks * num_servers)

        for task_idx in range(self.num_tasks):
            start_idx = task_idx * num_servers
            assigned_resource = random.randint(0, num_servers - 1)
            individual.CInd[start_idx + assigned_resource] = 1

        return individual

    def run(self):
        population = [self.schedule() for _ in range(self.population_size)]
        evaluated_population = self.fitness(population, self.data)
        
        def get_score(ind):
            if not ind.fitness or ind.fitness[0] == float('inf'): return float('inf')
            return sum(ind.fitness)

        best_overall = min(evaluated_population, key=get_score)
        return best_overall, evaluated_population