from config import *
import random
import numpy as np

class QIGA:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data
        self.distances = []

    # --- HELPER: Sorting & Dominance ---
    def non_dominated_sorting(self, population):
        fronts = [[]]
        for i, individual in enumerate(population):
            individual.domination_count = 0
            individual.dominated_set = []
            
            for j, other_individual in enumerate(population):
                if i == j:
                    continue
                if self.dominates(individual.fitness, other_individual.fitness):
                    individual.dominated_set.append(other_individual)
                elif self.dominates(other_individual.fitness, individual.fitness):
                    individual.domination_count += 1
                    
            if individual.domination_count == 0:
                individual.rank = 0
                fronts[0].append(individual)
        
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for individual in fronts[current_front]:
                for dominated_individual in individual.dominated_set:
                    dominated_individual.domination_count -= 1
                    if dominated_individual.domination_count == 0:
                        dominated_individual.rank = current_front + 1
                        next_front.append(dominated_individual)
            current_front += 1
            fronts.append(next_front)
        
        fronts.pop()
        return fronts

    def dominates(self, fitness1, fitness2):
        return all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))

    def calculate_crowding_distance(self, front):
        if not front: return
        num_objectives = len(front[0].fitness)
        for individual in front:
            individual.crowding_distance = 0
        
        for i in range(num_objectives):
            front.sort(key=lambda ind: ind.fitness[i])
            if not front: continue
            
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            min_fitness = front[0].fitness[i]
            max_fitness = front[-1].fitness[i]
            
            if max_fitness == min_fitness: continue

            for j in range(1, len(front) - 1):
                front[j].crowding_distance += (front[j + 1].fitness[i] - front[j - 1].fitness[i]) / (max_fitness - min_fitness)

    # --- QUANTUM CORE FUNCTIONS ---

    def _initialize_population(self):
        users = self.data['User'].all() 
        edge_servers = self.data['EdgeServer'].all()
        
        population = []
        
        for _ in range(self.population_size):
            individual = Individual()
            individual.QInd = []
            # Initialize random qubits
            for _ in range(len(users) * len(edge_servers)):
                theta_ij = random.uniform(0, np.pi)
                q_ij = np.array([[np.cos(theta_ij)], [np.sin(theta_ij)]])
                individual.QInd.append(q_ij)
            population.append(individual)

        return population

    def _quantum_observation(self, population):
        for individual in population:
            classical_individual = []
            # Process each task's slice of qubits
            num_servers = self.data['EdgeServer'].count()
            
            for i in range(0, len(individual.QInd), num_servers):
                task_qubits = individual.QInd[i:i + num_servers]
                
                # --- FIX: Calculate probability using Magnitude Squared (Absolute value) ---
                # This handles Complex numbers from Phase Gates correctly.
                probabilities = np.array([np.abs(q[1][0])**2 for q in task_qubits]).flatten()
                # -------------------------------------------------------------------------
                
                # Normalize probabilities to sum to 1
                if probabilities.sum() == 0:
                    probabilities = np.ones(len(probabilities)) / len(probabilities)
                else:
                    probabilities = probabilities / probabilities.sum()

                # Collapse to Classical State
                if np.random.rand() < 0.9:
                    # Greedy choice based on probability
                    selected_resource = np.argmax(probabilities)
                else:
                    # Random choice exploration
                    selected_resource = np.random.choice(len(probabilities), p=probabilities)
                
                # One-hot encode the selection
                classical_value = [1 if j == selected_resource else 0 for j in range(num_servers)]
                classical_individual.extend(classical_value)

            individual.CInd = classical_individual

        return population

    def _quantum_cnot_gate(self, target, control):
        # --- FIX: Use Magnitude Squared for Control check ---
        # If control qubit has high prob of being |1>
        if np.abs(control[1][0])**2 > 0.5:
            # Swap alpha and beta (Not Gate / Pauli-X)
            target = np.array([[target[1][0]], [target[0][0]]])
        return target

    def _quantum_phase_gate(self, qubit, phase):
        phase_matrix = np.array([[np.exp(1j * phase), 0], [0, np.exp(-1j * phase)]])
        return np.dot(phase_matrix, qubit)

    def _quantum_mutation(self, individual, generation):
        mutation_rate = 0.1
        for i in range(len(individual.QInd)):
            if np.random.rand() < mutation_rate:
                # CNOT Mutation
                control_qubit = individual.QInd[i]
                if i + 1 < len(individual.QInd):
                    target_qubit = individual.QInd[i + 1]
                    individual.QInd[i + 1] = self._quantum_cnot_gate(target_qubit, control_qubit)

                # Phase Rotation Mutation
                phase = (np.pi / 4) * (generation / self.generation_count)
                individual.QInd[i] = self._quantum_phase_gate(control_qubit, phase)
        return individual

    def _quantum_tournament_selection(self, population, tournament_size=3):
        tournament = np.random.choice(population, tournament_size, replace=False)
        sorted_tournament = sorted(tournament, key=lambda ind: (ind.rank, -ind.crowding_distance))
        return sorted_tournament[0]

    # --- ADAPTIVE CROSSOVER ---
    def _quantum_crossover(self, parent1, parent2, current_gen, max_generations, crossover_rate=0.8):
        offspring1 = Individual()
        offspring2 = Individual()
        offspring1.QInd = []
        offspring2.QInd = []

        if np.random.rand() < crossover_rate:
            # Adaptive Rotation Logic
            max_theta = 0.1 * np.pi  
            min_theta = 0.01 * np.pi 
            
            progress = current_gen / max_generations
            decay = 1 - progress
            theta_c = min_theta + (max_theta - min_theta) * decay

            R_theta_c = np.array([[np.cos(theta_c), -np.sin(theta_c)], [np.sin(theta_c), np.cos(theta_c)]])
            R_theta_nc = np.array([[np.cos(-theta_c), -np.sin(-theta_c)], [np.sin(-theta_c), np.cos(-theta_c)]])
            
            for i in range(len(parent1.QInd)):
                # Apply rotation matrix
                q1 = np.dot(R_theta_c, parent1.QInd[i])
                q2 = np.dot(R_theta_nc, parent2.QInd[i])
                offspring1.QInd.append(q1)
                offspring2.QInd.append(q2)
        else:
            offspring1.QInd = [q.copy() for q in parent1.QInd]
            offspring2.QInd = [q.copy() for q in parent2.QInd]

        return offspring1, offspring2

    def _quantum_offspring_generation(self, population, generation):
        fronts = self.non_dominated_sorting(population)
        for front in fronts:
            self.calculate_crowding_distance(front)

        selection_pool = [ind for front in fronts for ind in front]

        offspring_population = []
        while len(offspring_population) < self.population_size:
            parent1 = self._quantum_tournament_selection(selection_pool)
            parent2 = self._quantum_tournament_selection(selection_pool)
            
            child1, child2 = self._quantum_crossover(parent1, parent2, generation, self.generation_count)
            
            child1 = self._quantum_mutation(child1, generation)
            child2 = self._quantum_mutation(child2, generation)
            
            offspring_population.append(child1)
            if len(offspring_population) < self.population_size:
                offspring_population.append(child2)
        
        return offspring_population

    def _quantum_elitism_selection(self, population, new_population, size):
        combined_population = population + new_population
        fronts = self.non_dominated_sorting(combined_population)

        selected_population = []
        for front in fronts:
            self.calculate_crowding_distance(front)
            front.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))
            
            if len(selected_population) + len(front) <= size:
                selected_population.extend(front)
            else:
                remaining = size - len(selected_population)
                selected_population.extend(front[:remaining])
                break
        
        return selected_population

    def run(self):
        population = self._initialize_population()
        population = self._quantum_observation(population)
        population = self.fitness(population, self.data)

        for i in range(self.generation_count):
            new_population = self._quantum_offspring_generation(population, i)
            new_population = self._quantum_observation(new_population)
            new_population = self.fitness(new_population, self.data)
            population = self._quantum_elitism_selection(population, new_population, self.population_size)
        
        return population