from config import *

class RR:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data
        self.current_resource_idx = 0
        
        self.all_tasks = get_all_tasks(data)
        self.num_tasks = len(self.all_tasks)

    def schedule(self):
        individual = Individual()
        num_servers = self.data['EdgeServer'].count()
        individual.CInd = [0] * (self.num_tasks * num_servers)

        for task_idx in range(self.num_tasks):
            start_idx = task_idx * num_servers
            assigned_resource = self.current_resource_idx % num_servers
            
            individual.CInd[start_idx + assigned_resource] = 1
            self.current_resource_idx += 1

        return individual

    def run(self):
        # Round Robin is deterministic for a given starting index. 
        population = [self.schedule()]
        evaluated_population = self.fitness(population, self.data)
        return evaluated_population[0], evaluated_population