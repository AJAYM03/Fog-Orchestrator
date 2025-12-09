from config import *
import random
import numpy as np
import copy

class HybridQIGA:
    def __init__(self, fitness, population_size, generation_count, data):
        self.fitness = fitness
        self.population_size = population_size
        self.generation_count = generation_count
        self.data = data
        self.num_tasks = self.data['User'].count()
        self.num_resources = self.data['EdgeServer'].count()
        
        # Quantum Parameters
        # Rotation angle delta (Learning Rate)
        self.theta = 0.05 * np.pi 
        
        # Pre-fetch static data for the Smart Repair heuristic to run fast
        self.users = self.data['User'].all()
        self.servers = self.data['EdgeServer'].all()
        
        # Cache Server Frequencies (Hz)
        self.server_freqs = []
        for s in self.servers:
            # Use the helper from config.py
            f = get_freq(s.model_name, s)
            self.server_freqs.append(f)
            
        # Cache Task Weights (Instructions)
        self.task_weights = []
        for u in self.users:
            # Assuming 1 app -> 1 service per user as per config logic
            t = u.applications[0].services[0]
            self.task_weights.append(t.weight)

    # --- NSGA-II Helpers (Pareto Optimization) ---
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
            
            min_fitness = front[0].fitness[m]
            max_fitness = front[-1].fitness[m]
            
            if max_fitness == min_fitness: continue
            
            # --- FIX: Variable names match now ---
            norm = max_fitness - min_fitness
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i+1].fitness[m] - front[i-1].fitness[m]) / norm

    # --- Quantum Operations ---

    def _initialize_population(self):
        """
        Initializes the Quantum 'Mind' (Q-Individual).
        We only need ONE Q-Individual which represents the probability 
        distribution of the entire population.
        """
        num_qubits = self.num_tasks * self.num_resources
        q_ind = []
        for _ in range(num_qubits):
            # Start in Superposition: alpha = 1/sqrt(2), beta = 1/sqrt(2)
            # This means 50% chance for 0 or 1.
            q_ind.append(np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]]))
        return q_ind

    def _measure(self, q_ind):
        """
        Collapses the Quantum State to create a population of Classical Solutions.
        """
        classical_pop = []
        
        for _ in range(self.population_size):
            ind = Individual()
            ind.CInd = []
            
            # Process block of qubits for each task
            for i in range(0, len(q_ind), self.num_resources):
                task_qubits = q_ind[i : i + self.num_resources]
                
                # Calculate probability of each server (|beta|^2)
                probs = np.array([np.abs(q[1][0])**2 for q in task_qubits]).flatten()
                
                # Normalize probabilities to sum to 1
                if probs.sum() == 0: 
                    probs = np.ones(len(probs)) / len(probs)
                else: 
                    probs = probs / probs.sum()
                
                # Selection (Observation)
                chosen_server = np.random.choice(len(probs), p=probs)
                
                # Create One-Hot gene
                gene = [0] * self.num_resources
                gene[chosen_server] = 1
                ind.CInd.extend(gene)
                
            classical_pop.append(ind)
        return classical_pop

    def _update_quantum_gates(self, q_ind, best_solution):
        """
        Steering: Rotates the Qubits towards the Best Solution found.
        This provides the 'Evolutionary Pressure'.
        """
        if best_solution is None:
            return q_ind

        for i in range(len(q_ind)):
            # What did the best solution have at this bit? 0 or 1?
            target_bit = best_solution.CInd[i]
            
            alpha = q_ind[i][0][0]
            beta = q_ind[i][1][0]
            
            # Determine Rotation Direction
            # If target is 1, we rotate towards |1> (increase beta)
            # If target is 0, we rotate towards |0> (increase alpha)
            
            direction = 0
            # Logic table for rotation direction to converge to target_bit
            if target_bit == 1:
                # We want to increase probability of 1.
                # If signs are same, rotate one way; if diff, rotate other.
                direction = 1 if alpha * beta > 0 else -1 
            else:
                # We want to increase probability of 0.
                direction = -1 if alpha * beta > 0 else 1
            
            rotation_angle = direction * self.theta
            
            # Rotation Matrix
            rot = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle), np.cos(rotation_angle)]])
            
            # Apply Gate
            q_ind[i] = np.dot(rot, q_ind[i])
            
        return q_ind

    # --- SMART REPAIR MECHANISM (Utilization-Based) ---
    
    def repair_population(self, population):
        """
        Heuristic Step: Inspects server utilization (Load / Frequency) and performs
        smart swaps to alleviate bottlenecks.
        
        Fixes:
        1. ComputeHeavy: Moves tasks from slow servers to fast servers.
        2. HighLoad: Balances processing time, not just task counts.
        """
        for ind in population:
            # 1. Calculate Estimated Processing Load per Server
            # Unit: Seconds of estimated CPU time
            server_processing_loads = [0.0] * self.num_resources
            task_assignments = [-1] * self.num_tasks
            
            for t in range(self.num_tasks):
                start = t * self.num_resources
                end = start + self.num_resources
                segment = ind.CInd[start:end]
                
                try:
                    s_idx = segment.index(1)
                    task_assignments[t] = s_idx
                    
                    # LOAD CALCULATION: Weight / Frequency
                    # This tells us how 'busy' the server really is, unlike simple task counts.
                    if self.server_freqs[s_idx] > 0:
                        load_impact = self.task_weights[t] / self.server_freqs[s_idx]
                        server_processing_loads[s_idx] += load_impact
                    else:
                        # Fallback for zero freq (shouldn't happen)
                        server_processing_loads[s_idx] += 1e9 
                        
                except ValueError:
                    pass

            # 2. Smart Repair Logic
            # Find the most bottlenecked server (Highest Processing Load)
            max_load = max(server_processing_loads)
            min_load = min(server_processing_loads)
            
            # Threshold: Only move if the imbalance is significant
            # e.g., Max load is 20% higher than Min load
            if max_load > min_load * 1.2:
                
                # Identify Source (Bottleneck) and Target (Idle/Fast)
                max_s_idx = server_processing_loads.index(max_load)
                min_s_idx = server_processing_loads.index(min_load)
                
                # Get all tasks currently on the bottleneck server
                tasks_on_bottleneck = [t for t, s in enumerate(task_assignments) if s == max_s_idx]
                
                if tasks_on_bottleneck:
                    # Pick a task to move. 
                    # Optimization: Pick a random one. Picking the largest might cause oscillation.
                    task_to_move = random.choice(tasks_on_bottleneck)
                    
                    # Apply Move in Genotype (CInd)
                    start = task_to_move * self.num_resources
                    
                    # Reset old server bit
                    ind.CInd[start + max_s_idx] = 0
                    # Set new server bit
                    ind.CInd[start + min_s_idx] = 1
                    
        return population

    # --- Main Loop ---

    def run(self):
        # 1. Initialize Quantum State
        q_ind = self._initialize_population()
        
        best_overall = None
        
        for _ in range(self.generation_count):
            # 2. Measurement (Quantum -> Classical)
            classical_pop = self._measure(q_ind)
            
            # 3. Smart Hybrid Repair (Heuristic Improvement)
            # This fixes bottlenecks before fitness evaluation
            classical_pop = self.repair_population(classical_pop)
            
            # 4. Evaluation
            classical_pop = self.fitness(classical_pop, self.data)
            
            # 5. Elitism & Best Selection (NSGA-II Logic)
            fronts = self.non_dominated_sorting(classical_pop)
            self.calculate_crowding_distance(fronts[0])
            fronts[0].sort(key=lambda x: x.crowding_distance, reverse=True)
            
            best_current = fronts[0][0]
            
            # Update Global Best
            if best_overall is None:
                best_overall = copy.deepcopy(best_current)
            elif self.dominates(best_current.fitness, best_overall.fitness):
                best_overall = copy.deepcopy(best_current)
            elif not self.dominates(best_overall.fitness, best_current.fitness):
                # If mutually non-dominated, occasionally switch to maintain diversity
                if random.random() < 0.3:
                    best_overall = copy.deepcopy(best_current)

            # 6. Quantum Update (Steer Qubits towards Best)
            q_ind = self._update_quantum_gates(q_ind, best_overall)

        return classical_pop