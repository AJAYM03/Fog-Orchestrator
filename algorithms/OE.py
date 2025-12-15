import random
from config import *

class OE:
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

        edge_server_indices = []
        all_servers = self.data['EdgeServer'].all()
        for i, s in enumerate(all_servers):
            if "Cloud" not in s.model_name:
                edge_server_indices.append(i)
        
        if not edge_server_indices:
            edge_server_indices = range(num_servers)

        for i in range(self.num_tasks):
            chosen_server_idx = random.choice(edge_server_indices)
            gene_start = i * num_servers
            bit_pos = gene_start + chosen_server_idx
            individual.CInd[bit_pos] = 1
            
        return individual

    def run(self):
        population = [self.schedule() for _ in range(self.population_size)]
        evaluated_population = self.fitness(population, self.data)
        
        # Helper to get score for sorting
        def get_score(ind):
            if not ind.fitness or ind.fitness[0] == float('inf'): return float('inf')
            return sum(ind.fitness)

        best_overall = min(evaluated_population, key=get_score)
        return best_overall, evaluated_population