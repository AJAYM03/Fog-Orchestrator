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
        
        # Robust Task Counting
        self.all_tasks = get_all_tasks(data)
        self.num_tasks = len(self.all_tasks)
        self.num_resources = self.data['EdgeServer'].count()
        
        self.gene_size = self.num_tasks * self.num_resources

    # --- Multi-Objective Helpers (NSGA-II Logic) ---
    def dominates(self, fitness1, fitness2):
        """Return True if fitness1 dominates fitness2."""
        return all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))

    def non_dominated_sorting(self, population):
        """Assigns rank to each individual based on Pareto dominance."""
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_set = []
            for q in population:
                if self.dominates(p.fitness, q.fitness):
                    p.dominated_set.append(q)
                elif self.dominates(q.fitness, p.fitness):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        if not fronts[-1]:
            fronts.pop()
            
        return fronts

    def calculate_crowding_distance(self, front):
        """Assigns crowding distance to maintain diversity."""
        if not front: return
        num_objectives = len(front[0].fitness)
        
        for p in front:
            p.crowding_distance = 0
            
        for m in range(num_objectives):
            front.sort(key=lambda x: x.fitness[m])
            
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            min_fit = front[0].fitness[m]
            max_fit = front[-1].fitness[m]
            
            if max_fit == min_fit: continue
            
            norm = max_fit - min_fit
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i+1].fitness[m] - front[i-1].fitness[m]) / norm

    # --- Core GA Methods ---

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

    def selection(self, population):
        # Tournament Selection based on (Rank, -CrowdingDistance)
        competitors = random.sample(population, 2)
        best = competitors[0]
        for comp in competitors[1:]:
            if comp.rank < best.rank:
                best = comp
            elif comp.rank == best.rank:
                if comp.crowding_distance > best.crowding_distance:
                    best = comp
        return best

    def crossover(self, p1, p2):
        # Task-Aware Uniform Crossover
        c1, c2 = Individual(), Individual()
        c1.CInd = []
        c2.CInd = []
        
        for i in range(self.num_tasks):
            start = i * self.num_resources
            end = start + self.num_resources
            
            # 50% chance to swap this task assignment between parents
            if random.random() < 0.5:
                c1.CInd.extend(p1.CInd[start:end])
                c2.CInd.extend(p2.CInd[start:end])
            else:
                c1.CInd.extend(p2.CInd[start:end])
                c2.CInd.extend(p1.CInd[start:end])
        
        return c1, c2

    def mutation(self, individual):
        # Safety check for empty tasks
        if self.num_tasks == 0:
            return individual

        # Task-Aware Mutation
        mutation_rate = 1.0 / self.num_tasks
        new_genes = individual.CInd[:] 
        
        if random.random() < 0.5: 
            # Pick one random task to mutate
            task_idx = random.randint(0, self.num_tasks - 1)
            new_server = random.randint(0, self.num_resources - 1)
            
            start = task_idx * self.num_resources
            end = start + self.num_resources
            
            # Reset this task's bits to 0
            new_genes[start:end] = [0] * self.num_resources
            # Set new server bit to 1
            new_genes[start + new_server] = 1
            
        individual.CInd = new_genes
        return individual

    def run(self):
        population = self.initialize_population()
        population = self.fitness(population, self.data)
        
        fronts = self.non_dominated_sorting(population)
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        for _ in range(self.generation_count):
            offspring = []
            while len(offspring) < self.population_size:
                p1 = self.selection(population)
                p2 = self.selection(population)
                c1, c2 = self.crossover(p1, p2)
                offspring.extend([self.mutation(c1), self.mutation(c2)])
            
            offspring = self.fitness(offspring, self.data)
            
            combined = population + offspring
            fronts = self.non_dominated_sorting(combined)
            
            new_population = []
            for front in fronts:
                self.calculate_crowding_distance(front)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    remaining = self.population_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break
            
            population = new_population
            
        # Ensure best is at index 0 (population is sorted by NSGA-II logic already)
        best_overall = population[0] if population else None
        return best_overall, population