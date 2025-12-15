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
        
        # Robust Task Counting
        self.all_tasks = get_all_tasks(data)
        self.num_tasks = len(self.all_tasks)
        self.num_resources = self.data['EdgeServer'].count()
        
        # Quantum Parameters
        self.theta = 0.05 * np.pi 
        
        # Cache Server Data
        self.servers = self.data['EdgeServer'].all()
        self.server_freqs = []
        self.server_costs = []
        
        for s in self.servers:
            f = get_freq(s.model_name, s)
            self.server_freqs.append(f)
            c = s.power_model_parameters.get('monetary_cost', 0)
            self.server_costs.append(c)
            
        # Cache Task Weights and Calculate Average
        self.task_weights = []
        for t_dict in self.all_tasks:
            self.task_weights.append(t_dict['service'].weight)
            
        self.avg_task_weight = sum(self.task_weights) / len(self.task_weights) if self.task_weights else 0

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
        num_objectives = len(front[0].fitness)
        for p in front: p.crowding_distance = 0
        for m in range(num_objectives):
            front.sort(key=lambda x: x.fitness[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            min_v, max_v = front[0].fitness[m], front[-1].fitness[m]
            if max_v == min_v: continue
            norm = max_v - min_v
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i+1].fitness[m] - front[i-1].fitness[m]) / norm

    # --- Quantum Operations ---

    def _initialize_population(self):
        num_qubits = self.num_tasks * self.num_resources
        q_ind = []
        for _ in range(num_qubits):
            q_ind.append(np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]]))
        return q_ind

    def _measure(self, q_ind):
        classical_pop = []
        for _ in range(self.population_size):
            ind = Individual()
            ind.CInd = []
            for i in range(0, len(q_ind), self.num_resources):
                task_qubits = q_ind[i : i + self.num_resources]
                probs = np.array([np.abs(q[1][0])**2 for q in task_qubits]).flatten()
                if probs.sum() == 0: probs = np.ones(len(probs)) / len(probs)
                else: probs = probs / probs.sum()
                chosen_server = np.random.choice(len(probs), p=probs)
                gene = [0] * self.num_resources
                gene[chosen_server] = 1
                ind.CInd.extend(gene)
            classical_pop.append(ind)
        return classical_pop

    def _update_quantum_gates(self, q_ind, best_solution):
        if best_solution is None: return q_ind
        for i in range(len(q_ind)):
            target_bit = best_solution.CInd[i]
            alpha = q_ind[i][0][0]
            beta = q_ind[i][1][0]
            
            if target_bit == 1 and abs(beta)**2 > 0.99: continue
            if target_bit == 0 and abs(alpha)**2 > 0.99: continue

            direction = 0
            if target_bit == 1:
                direction = 1 if alpha * beta > 0 else -1 
            else:
                direction = -1 if alpha * beta > 0 else 1
            
            rotation_angle = direction * self.theta
            rot = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle), np.cos(rotation_angle)]])
            q_ind[i] = np.dot(rot, q_ind[i])
            
            norm = np.linalg.norm(q_ind[i])
            q_ind[i] = q_ind[i] / norm
            
        return q_ind

    # --- POLYMORPHIC REPAIR MECHANISM ---
    
    def repair_population(self, population):
        for ind in population:
            server_processing_loads = [0.0] * self.num_resources
            task_assignments = [-1] * self.num_tasks
            
            for t in range(self.num_tasks):
                start = t * self.num_resources
                end = start + self.num_resources
                segment = ind.CInd[start:end]
                try:
                    s_idx = segment.index(1)
                    task_assignments[t] = s_idx
                    if self.server_freqs[s_idx] > 0:
                        load_impact = self.task_weights[t] / self.server_freqs[s_idx]
                        server_processing_loads[s_idx] += load_impact
                    else:
                        server_processing_loads[s_idx] += 1e9 
                except ValueError:
                    pass

            max_load = max(server_processing_loads)
            max_s_idx = server_processing_loads.index(max_load)
            
            tasks_on_bottleneck = [t for t, s in enumerate(task_assignments) if s == max_s_idx]
            
            if tasks_on_bottleneck:
                task_to_move = random.choice(tasks_on_bottleneck)
                weight = self.task_weights[task_to_move]
                best_target = -1
                
                if weight > self.avg_task_weight * 1.2:
                    candidates = [s for s in range(self.num_resources) if s != max_s_idx]
                    if candidates:
                        best_target = max(candidates, key=lambda s: self.server_freqs[s])
                else:
                    candidates = [s for s in range(self.num_resources) 
                                  if s != max_s_idx and server_processing_loads[s] < max_load]
                    if candidates:
                        best_target = min(candidates, key=lambda s: self.server_costs[s])
                
                if best_target != -1:
                    start = task_to_move * self.num_resources
                    ind.CInd[start + max_s_idx] = 0
                    ind.CInd[start + best_target] = 1
                        
        return population

    # --- Main Loop ---

    def run(self):
        q_ind = self._initialize_population()
        best_overall = None
        classical_pop = []
        
        for _ in range(self.generation_count):
            classical_pop = self._measure(q_ind)
            classical_pop = self.repair_population(classical_pop)
            classical_pop = self.fitness(classical_pop, self.data)
            
            fronts = self.non_dominated_sorting(classical_pop)
            self.calculate_crowding_distance(fronts[0])
            fronts[0].sort(key=lambda x: x.crowding_distance, reverse=True)
            best_current = fronts[0][0]
            
            if best_overall is None:
                best_overall = copy.deepcopy(best_current)
            elif self.dominates(best_current.fitness, best_overall.fitness):
                best_overall = copy.deepcopy(best_current)
            elif not self.dominates(best_overall.fitness, best_current.fitness):
                if random.random() < 0.3:
                    best_overall = copy.deepcopy(best_current)

            q_ind = self._update_quantum_gates(q_ind, best_overall)

        if best_overall:
            if best_overall not in classical_pop:
                classical_pop.append(best_overall)

        return best_overall, classical_pop