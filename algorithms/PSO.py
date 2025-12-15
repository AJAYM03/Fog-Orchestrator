from config import *
import random
import numpy as np
import copy

class PSO:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data
        
        # Robust Task Counting
        self.all_tasks = get_all_tasks(data)
        self.num_tasks = len(self.all_tasks)
        self.num_resources = self.data['EdgeServer'].count()
        
        self.w = 0.5 
        self.c1 = 1.5 
        self.c2 = 1.5 

    def get_score(self, individual):
        if not individual.fitness or individual.fitness[0] == float('inf'): return float('inf')
        return sum(individual.fitness) 

    def discrete_to_continuous(self, individual):
        return np.array(individual.CInd, dtype=float)

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

    def run(self):
        particles = []
        velocities = []
        p_best = []
        p_best_scores = []
        
        for _ in range(self.population_size):
            ind = Individual()
            ind.CInd = []
            for _ in range(self.num_tasks):
                gene = [0] * self.num_resources
                gene[random.randint(0, self.num_resources - 1)] = 1
                ind.CInd.extend(gene)
            particles.append(ind)
            velocities.append(np.random.uniform(-1, 1, len(ind.CInd)))
        
        particles = self.fitness(particles, self.data)
        p_best = copy.deepcopy(particles)
        p_best_scores = [self.get_score(p) for p in particles]
        
        g_best = min(p_best, key=self.get_score)
        g_best_score = self.get_score(g_best)
        g_best_pos = self.discrete_to_continuous(g_best)

        for _ in range(self.generation_count):
            new_particles = []
            
            for i in range(self.population_size):
                current_pos = self.discrete_to_continuous(particles[i])
                p_best_pos = self.discrete_to_continuous(p_best[i])
                
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (p_best_pos - current_pos) + 
                                 self.c2 * r2 * (g_best_pos - current_pos))
                
                new_pos_continuous = current_pos + velocities[i]
                
                new_ind = Individual()
                new_ind.CInd = self.continuous_to_discrete(new_pos_continuous)
                new_particles.append(new_ind)

            new_particles = self.fitness(new_particles, self.data)
            
            for i in range(self.population_size):
                score = self.get_score(new_particles[i])
                
                if score < p_best_scores[i]:
                    p_best[i] = copy.deepcopy(new_particles[i])
                    p_best_scores[i] = score
                    
                    if score < g_best_score:
                        g_best = copy.deepcopy(new_particles[i])
                        g_best_score = score
                        g_best_pos = self.discrete_to_continuous(g_best)
            
            particles = new_particles

        return g_best, particles