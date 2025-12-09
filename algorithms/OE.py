import random
from config import *

# --- FIX: Removed duplicate 'class Individual' definition. ---
# It now correctly uses the full class from config.py

class OE:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data

    def schedule(self):
        # This now uses the robust Individual class from config.py
        individual = Individual()

        # Calculate number of bits needed
        num_users = self.data['User'].count()
        num_servers = self.data['EdgeServer'].count()
        
        # Initialize CInd with zeros
        individual.CInd = [0] * (num_users * num_servers)

        # OE Logic: Identify "Edge" servers (exclude Cloud)
        edge_server_indices = []
        all_servers = self.data['EdgeServer'].all()
        for i, s in enumerate(all_servers):
            # Assume Cloud has "Cloud" in name
            if "Cloud" not in s.model_name:
                edge_server_indices.append(i)
        
        # Safety fallback
        if not edge_server_indices:
            edge_server_indices = range(num_servers)

        # Assign each user to a random EDGE server
        for i in range(num_users):
            chosen_server_idx = random.choice(edge_server_indices)
            
            # Calculate One-Hot position
            gene_start = i * num_servers
            bit_pos = gene_start + chosen_server_idx
            
            individual.CInd[bit_pos] = 1
            
        return individual

    def run(self):
        population = [self.schedule() for _ in range(self.population_size)]
        evaluated_population = self.fitness(population, self.data)
        return evaluated_population