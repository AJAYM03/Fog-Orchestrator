import random
import numpy as np
from collections import deque

# Simulation Parameters
K_POP_SIZE = 32
K_GEN_SIZE = 60

# --- OPTIMIZATION WEIGHTS ---
# RECOMMENDED BALANCED SET for "Smart" Repair validation
# We punish Cost (0.5) heavily because the Repair logic will now protecting Latency (0.4)
W_ENERGY = 0.1
W_LATENCY = 0.4
W_COST = 0.5 

class Individual:
    def __init__(self):
        self.QInd = []
        self.CInd = []
        self.fitness = [] 
        self.cost = float('inf')
        self.energy = float('inf')
        self.latency = float('inf')
        self.crowding_distance = float('inf')
        self.rank = float('inf')
        self.qos = 0
        self.resource_utilization = 0
        self.missed_deadlines = 0
        self.max_resource_latency = 0
        self.mem_overload = False

def get_freq(model_name, resource=None):
    if resource and hasattr(resource, 'frequency') and resource.frequency > 0:
        return resource.frequency
    if "E5430" in model_name: return 2.66 * 10e9
    elif "Raspberry" in model_name: return 1.8 * 10e9
    elif "Cloud" in model_name: return 50.0 * 10e9
    return 2.0 * 10e9 

def get_all_tasks(data):
    """
    Helper to flatten all tasks (services) from all users.
    Returns a list of dicts: [{'user': user_obj, 'service': service_obj}, ...]
    This ensures consistent ordering for gene mapping.
    """
    all_tasks = []
    for user in data['User'].all():
        for app in user.applications:
            for service in app.services:
                all_tasks.append({'user': user, 'service': service})
    return all_tasks

def decode(data, individual):
    """
    Decodes an individual's CInd (One-Hot) into a mapping of resources to tasks.
    Returns: {EdgeServer: [{'user': u, 'service': s}, ...]}
    """
    num_resources = data['EdgeServer'].count()
    all_tasks = get_all_tasks(data)
    num_tasks = len(all_tasks)
    
    resources_map = {r: [] for r in data['EdgeServer'].all()}
    
    # Safety check for gene length
    if len(individual.CInd) < num_tasks * num_resources: 
        return resources_map 
        
    for i in range(num_tasks):
        start_index = i * num_resources
        end_index = start_index + num_resources
        
        if start_index < len(individual.CInd):
            slice_data = individual.CInd[start_index:end_index]
            # Verify exactly one server is selected (One-Hot)
            if sum(slice_data) > 0:
                assigned_resource_index = np.argmax(slice_data)
                target_server = data['EdgeServer'].all()[assigned_resource_index]
                resources_map[target_server].append(all_tasks[i])
                
    return resources_map

def memory_is_overloaded(task_dicts, av_memory):
    total_mem = sum(t['service'].memory_demand for t in task_dicts)
    return total_mem > av_memory

def get_exe_delay(av_frequency, task_weight):
    if av_frequency <= 0: return float('inf')
    return task_weight / av_frequency

def get_path_delay(resource_bs_id, user_bs_id, task_data_size, data, graph):
    if resource_bs_id == user_bs_id:
        user_bs = data['BaseStation'].find_by_id(user_bs_id)
        if user_bs and user_bs.wireless_delay > 0:
            return task_data_size / user_bs.wireless_delay
        return 0
        
    queue = deque([(resource_bs_id, 0)])
    visited = {resource_bs_id}
    
    while queue:
        current_node, cumulative_delay = queue.popleft()
        if current_node == user_bs_id:
            user_bs = data['BaseStation'].find_by_id(user_bs_id)
            wireless = (task_data_size / user_bs.wireless_delay) if user_bs.wireless_delay > 0 else 0
            return wireless + cumulative_delay
            
        for neighbor, bandwidth in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                link_delay = task_data_size / bandwidth if bandwidth > 0 else float('inf')
                queue.append((neighbor, cumulative_delay + link_delay))
                
    return float('inf')

def fitness(population, data):
    if not isinstance(population, list): population = [population]
    graph = data.get('graph', {})
    energy_values, latency_values, cost_values = [], [], []

    for individual in population:
        # Decode returns {Server: [List of Tasks]}
        resources_map = decode(data, individual)
        individual.missed_deadlines = 0
        individual.mem_overload = False
        total_energy = 0
        total_latency = 0
        total_cost = 0

        for resource, task_dicts in resources_map.items():
            resource_bs_id = resource.base_station.id
            av_frequency = get_freq(resource.model_name, resource)
            
            if memory_is_overloaded(task_dicts, resource.memory): 
                individual.mem_overload = True
            
            if task_dicts: 
                # Add base cost of the server if used
                total_cost += resource.power_model_parameters.get('monetary_cost', 0)

            # Sort tasks by deadline for simplistic scheduling assumption
            sorted_tasks = sorted(task_dicts, key=lambda t: t['service'].deadline)
            
            for t_dict in sorted_tasks:
                user = t_dict['user']
                task = t_dict['service']
                user_bs_id = user.base_station.id
                
                # --- PREDICTION LOGIC ENABLED ---
                # Simulate uncertainty in task weight estimation
                actual_task_weight = task.weight
                prediction_noise = random.uniform(0.9, 1.1) 
                predicted_task_weight = actual_task_weight * prediction_noise
                
                path_delay = get_path_delay(resource_bs_id, user_bs_id, task.data_size, data, graph)
                exe_delay = get_exe_delay(av_frequency, predicted_task_weight) 
                delay = path_delay + exe_delay
                
                # Energy Calculation
                power_watts = resource.power_model_parameters.get('static_power_percentage', 0)
                energy_consumption = power_watts * exe_delay
                
                total_energy += energy_consumption
                total_latency += delay
                
                # Deadline Check
                if delay > task.deadline: 
                    individual.missed_deadlines += 1

        num_servers = data['EdgeServer'].count()
        # Average metrics per server or total? Using averages as per original design.
        individual.energy = total_energy / num_servers if num_servers > 0 else float('inf')
        individual.latency = total_latency / num_servers if num_servers > 0 else float('inf')
        individual.cost = total_cost 
        
        energy_values.append(individual.energy)
        latency_values.append(individual.latency)
        cost_values.append(individual.cost)

    if not energy_values: return population

    # Normalization
    min_e, max_e = min(energy_values), max(energy_values)
    min_l, max_l = min(latency_values), max(latency_values)
    min_c, max_c = min(cost_values), max(cost_values)

    for individual in population:
        norm_e = (individual.energy - min_e)/(max_e - min_e) if max_e > min_e else 0.0
        norm_l = (individual.latency - min_l)/(max_l - min_l) if max_l > min_l else 0.0
        norm_c = (individual.cost - min_c)/(max_c - min_c) if max_c > min_c else 0.0
        
        # Weighted penalty integration
        penalty = (individual.missed_deadlines * 1.0) + (100.0 if individual.mem_overload else 0.0)
        
        # --- APPLY WEIGHTS ---
        individual.fitness = [
            (norm_e * W_ENERGY) + penalty, 
            (norm_l * W_LATENCY) + penalty, 
            (norm_c * W_COST) + penalty
        ]

    return population