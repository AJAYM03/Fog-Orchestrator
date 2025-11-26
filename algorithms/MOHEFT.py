from config import *
import random
import copy

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
            # Random initialization (0 or 1)
            for _ in range(self.gene_size):
                individual.CInd.append(random.randint(0, 1))
            population.append(individual)
        return population

    # --- STANDARD GENETIC OPERATORS ---

    def tournament_selection(self, population):
        # Pick 2 random parents and choose the better one (Binary Tournament)
        competitors = random.sample(population, 2)
        
        # Selection based on Rank (lower is better) and Crowding Distance (higher is better)
        if competitors[0].rank < competitors[1].rank:
            return competitors[0]
        elif competitors[0].rank > competitors[1].rank:
            return competitors[1]
        else:
            # If ranks are equal, pick the one in the less crowded region
            if competitors[0].crowding_distance > competitors[1].crowding_distance:
                return competitors[0]
            else:
                return competitors[1]

    def crossover(self, parent1, parent2):
        # Uniform Crossover
        offspring1 = Individual()
        offspring2 = Individual()
        
        mask = [random.randint(0, 1) for _ in range(self.gene_size)]
        
        child1_genes = []
        child2_genes = []
        
        for i in range(self.gene_size):
            if mask[i] == 0:
                child1_genes.append(parent1.CInd[i])
                child2_genes.append(parent2.CInd[i])
            else:
                child1_genes.append(parent2.CInd[i])
                child2_genes.append(parent1.CInd[i])
                
        offspring1.CInd = child1_genes
        offspring2.CInd = child2_genes
        return offspring1, offspring2

    def mutation(self, individual):
        # Bit-Flip Mutation
        mutation_rate = 1.0 / self.gene_size # Standard heuristic
        
        new_genes = individual.CInd[:]
        for i in range(self.gene_size):
            if random.random() < mutation_rate:
                new_genes[i] = 1 - new_genes[i] # Flip 0->1 or 1->0
        
        individual.CInd = new_genes
        return individual

    def create_offspring(self, population):
        offspring_population = []
        
        # Create new children until we fill the population
        while len(offspring_population) < self.population_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            offspring_population.append(child1)
            if len(offspring_population) < self.population_size:
                offspring_population.append(child2)
                
        return offspring_population

    # --- NSGA-II SELECTION LOGIC (Same as QIGA) ---

    def non_dominated_sorting(self, population):
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
        fronts.pop()
        return fronts

    def dominates(self, fitness1, fitness2):
        # Strictly better in at least one, not worse in any
        return all(x <= y for x, y in zip(fitness1, fitness2)) and any(x < y for x, y in zip(fitness1, fitness2))

    def calculate_crowding_distance(self, front):
        if len(front) == 0: return
        num_objectives = len(front[0].fitness)
        
        for ind in front:
            ind.crowding_distance = 0
            
        for m in range(num_objectives):
            front.sort(key=lambda x: x.fitness[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            m_min = front[0].fitness[m]
            m_max = front[-1].fitness[m]
            
            if m_max == m_min: continue

            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i+1].fitness[m] - front[i-1].fitness[m]) / (m_max - m_min)

    def select_best_population(self, combined_population):
        fronts = self.non_dominated_sorting(combined_population)
        new_population = []
        
        for front in fronts:
            self.calculate_crowding_distance(front)
            
            # If adding this whole front fits, add it
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # Otherwise, sort by crowding distance and pick the best ones to fill up
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                needed = self.population_size - len(new_population)
                new_population.extend(front[:needed])
                break
                
        return new_population

    # --- MAIN RUN LOOP ---

    def run(self):
        # 1. Initialize
        population = self.initialize_population()
        population = self.fitness(population, self.data)

        # 2. Evolution Loop
        for _ in range(self.generation_count):
            # Create Offspring (Crossover & Mutation)
            offspring = self.create_offspring(population)
            offspring = self.fitness(offspring, self.data)
            
            # Combine Parents + Offspring
            combined_population = population + offspring
            
            # Survival of the Fittest (NSGA-II Selection)
            population = self.select_best_population(combined_population)
            
        return population