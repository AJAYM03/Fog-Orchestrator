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
        self.num_tasks = self.data['User'].count()
        self.num_resources = self.data['EdgeServer'].count()
        
        # PSO Parameters
        self.w = 0.5  # Inertia
        self.c1 = 1.5 # Cognitive (Personal Best)
        self.c2 = 1.5 # Social (Global Best)

    def get_score(self, individual):
        if not individual.fitness or individual.fitness[0] == float('inf'): return float('inf')
        return sum(individual.fitness) # Minimize sum of objectives

    def discrete_to_continuous(self, individual):
        # Convert binary CInd to continuous vector for PSO math
        return np.array(individual.CInd, dtype=float)

    def continuous_to_discrete(self, vector):
        # Convert continuous vector back to One-Hot CInd
        c_ind = []
        for i in range(self.num_tasks):
            start = i * self.num_resources
            end = start + self.num_resources
            segment = vector[start:end]
            
            # Argmax to pick the server
            chosen = np.argmax(segment)
            
            # One-hot encoding
            gene = [0] * self.num_resources
            gene[chosen] = 1
            c_ind.extend(gene)
        return c_ind

    def run(self):
        # 1. Initialize Particles
        particles = []
        velocities = []
        p_best = []
        p_best_scores = []
        
        # Initial Population
        for _ in range(self.population_size):
            ind = Individual()
            ind.CInd = []
            for _ in range(self.num_tasks):
                gene = [0] * self.num_resources
                gene[random.randint(0, self.num_resources - 1)] = 1
                ind.CInd.extend(gene)
            particles.append(ind)
            velocities.append(np.random.uniform(-1, 1, len(ind.CInd)))
        
        # Evaluate Initial
        particles = self.fitness(particles, self.data)
        p_best = copy.deepcopy(particles)
        p_best_scores = [self.get_score(p) for p in particles]
        
        # Find Global Best
        g_best = min(p_best, key=self.get_score)
        g_best_score = self.get_score(g_best)
        g_best_pos = self.discrete_to_continuous(g_best)

        # 2. Main Loop
        for _ in range(self.generation_count):
            new_particles = []
            
            for i in range(self.population_size):
                current_pos = self.discrete_to_continuous(particles[i])
                p_best_pos = self.discrete_to_continuous(p_best[i])
                
                # Update Velocity
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (p_best_pos - current_pos) + 
                                 self.c2 * r2 * (g_best_pos - current_pos))
                
                # Update Position
                new_pos_continuous = current_pos + velocities[i]
                
                # Create new Individual
                new_ind = Individual()
                new_ind.CInd = self.continuous_to_discrete(new_pos_continuous)
                new_particles.append(new_ind)

            # Evaluate
            new_particles = self.fitness(new_particles, self.data)
            
            # Update Bests
            for i in range(self.population_size):
                score = self.get_score(new_particles[i])
                
                # Update Personal Best
                if score < p_best_scores[i]:
                    p_best[i] = copy.deepcopy(new_particles[i])
                    p_best_scores[i] = score
                    
                    # Update Global Best
                    if score < g_best_score:
                        g_best = copy.deepcopy(new_particles[i])
                        g_best_score = score
                        g_best_pos = self.discrete_to_continuous(g_best)
            
            # Update population for next iter
            particles = new_particles

        # Return the final population (or list of g_best repeated, but population is standard)
        return particles