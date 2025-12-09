from config import *
import random
import numpy as np
import copy

class QIGA:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data
        # Rotation angle delta
        self.theta = 0.05 * np.pi 

    # --- NSGA-II Helpers ---
    def dominates(self, fitness1, fitness2):
        return all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))

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
        if not fronts[-1]: fronts.pop()
        return fronts

    def calculate_crowding_distance(self, front):
        if not front: return
        num_objs = len(front[0].fitness)
        for p in front: p.crowding_distance = 0
        for m in range(num_objs):
            front.sort(key=lambda x: x.fitness[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            min_v, max_v = front[0].fitness[m], front[-1].fitness[m]
            if max_v == min_v: continue
            for i in range(1, len(front)-1):
                front[i].crowding_distance += (front[i+1].fitness[m] - front[i-1].fitness[m]) / (max_v - min_v)

    # --- Quantum Operations ---

    def _initialize_population(self):
        # Create one Quantum Individual (The "Mind")
        # Standard QIGA evolves ONE probability vector, samples N times.
        num_qubits = self.data['User'].count() * self.data['EdgeServer'].count()
        q_ind = []
        for _ in range(num_qubits):
            # Start in Superposition (Equal probability)
            q_ind.append(np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]]))
        return q_ind

    def _measure(self, q_ind):
        # Generate classical population from Q-state
        classical_pop = []
        num_servers = self.data['EdgeServer'].count()
        
        for _ in range(self.population_size):
            ind = Individual()
            ind.CInd = []
            
            # Collapse Wavefunction
            # For scheduling, we process blocks of qubits per task
            for i in range(0, len(q_ind), num_servers):
                task_qubits = q_ind[i:i+num_servers]
                
                # Probabilities |beta|^2
                probs = np.array([np.abs(q[1][0])**2 for q in task_qubits]).flatten()
                
                # Normalize
                if probs.sum() == 0: probs = np.ones(len(probs))/len(probs)
                else: probs = probs / probs.sum()
                
                # Select Resource
                chosen = np.random.choice(len(probs), p=probs)
                
                # One-hot
                gene = [0]*num_servers
                gene[chosen] = 1
                ind.CInd.extend(gene)
                
            classical_pop.append(ind)
        return classical_pop

    def _update_quantum_gates(self, q_ind, best_solution):
        # Steer the Qubits towards the best solution
        for i in range(len(q_ind)):
            target_bit = best_solution.CInd[i] # 0 or 1
            
            alpha = q_ind[i][0][0]
            beta = q_ind[i][1][0]
            
            # Determine rotation direction
            # If target is 1, we want to increase beta (probability of 1)
            # If target is 0, we want to increase alpha (probability of 0)
            
            direction = 0
            if target_bit == 1:
                # Rotate towards |1>
                direction = 1 if alpha * beta > 0 else -1 
            else:
                # Rotate towards |0>
                direction = -1 if alpha * beta > 0 else 1
            
            theta = direction * self.theta
            
            # Rotation Matrix
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            
            q_ind[i] = np.dot(rot, q_ind[i])
            
        return q_ind

    def run(self):
        # 1. Initialize Quantum State (The "Mind")
        q_ind = self._initialize_population()
        
        best_overall = None
        
        for _ in range(self.generation_count):
            # 2. Measurement (Generate Classical Solutions)
            classical_pop = self._measure(q_ind)
            
            # 3. Evaluation
            classical_pop = self.fitness(classical_pop, self.data)
            
            # 4. Sorting to find Best
            fronts = self.non_dominated_sorting(classical_pop)
            self.calculate_crowding_distance(fronts[0])
            fronts[0].sort(key=lambda x: x.crowding_distance, reverse=True)
            
            best_current = fronts[0][0]
            
            # Update global best (Simple elitism)
            if best_overall is None or self.dominates(best_current.fitness, best_overall.fitness):
                best_overall = copy.deepcopy(best_current)
            elif self.dominates(best_overall.fitness, best_current.fitness):
                pass # Keep old best
            else:
                # Nondominated, maybe keep random or based on sparsity
                if random.random() < 0.5: best_overall = copy.deepcopy(best_current)

            # 5. Quantum Update (Steering)
            # Rotate Q-state towards the best solution found
            q_ind = self._update_quantum_gates(q_ind, best_overall)

        return classical_pop