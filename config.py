import random
import numpy as np
from collections import deque

# You can increase these to 64/100 for the "Grand Test" if you have time,
# but 32/60 is fine for testing the noise impact.
K_POP_SIZE = 32
K_GEN_SIZE = 60


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

def decode(data, individual):
    num_resources = data['EdgeServer'].count()
    num_users = data['User'].count()
    resources_and_users = {r: [] for r in data['EdgeServer'].all()}
    if len(individual.CInd) < num_users * num_resources: return resources_and_users
    for i in range(num_users):
        start_index = i * num_resources
        end_index = start_index + num_resources
        if start_index < len(individual.CInd):
            slice_data = individual.CInd[start_index:end_index]
            if sum(slice_data) > 0:
                assigned_resource_index = np.argmax(slice_data)
                resources_and_users[data['EdgeServer'].all()[assigned_resource_index]].append(data['User'].all()[i])
    return resources_and_users

def memory_is_overloaded(users, av_memory):
    total_mem = sum(u.applications[0].services[0].memory_demand for u in users)
    return total_mem > av_memory

def get_exe_delay(av_frequency, task_weight):
    return task_weight / av_frequency

def get_path_delay(resource_bs_id, user_bs_id, task_data_size, data, user, graph):
    if resource_bs_id == user_bs_id:
        user_bs = data['BaseStation'].find_by_id(user_bs_id)
        return task_data_size / user_bs.wireless_delay
    queue = deque([(resource_bs_id, 0)])
    visited = {resource_bs_id}
    while queue:
        current_node, cumulative_delay = queue.popleft()
        if current_node == user_bs_id:
            user_bs = data['BaseStation'].find_by_id(user_bs_id)
            return (task_data_size / user_bs.wireless_delay) + cumulative_delay
        for neighbor, bandwidth in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                link_delay = task_data_size / bandwidth
                queue.append((neighbor, cumulative_delay + link_delay))
    return float('inf')

def fitness(population, data):
    if not isinstance(population, list): population = [population]
    graph = data.get('graph', {})
    energy_values, latency_values, cost_values = [], [], []

    for individual in population:
        resources_and_users = decode(data, individual)
        individual.missed_deadlines = 0
        individual.mem_overload = False
        total_energy = 0
        total_latency = 0
        total_cost = 0

        for resource, users in resources_and_users.items():
            resource_bs_id = resource.base_station.id
            av_frequency = get_freq(resource.model_name, resource)
            if memory_is_overloaded(users, resource.memory): individual.mem_overload = True
            if users: total_cost += resource.power_model_parameters.get('monetary_cost', 0)

            sorted_users = sorted(users, key=lambda u: u.applications[0].services[0].deadline)
            for user in sorted_users:
                user_bs_id = user.base_station.id
                task = user.applications[0].services[0]
                
                # --- PREDICTION LOGIC ENABLED ---
                actual_task_weight = task.weight
                
                # 1. Generate Noise (+/- 10% uncertainty)
                prediction_noise = random.uniform(0.9, 1.1) 
                
                # 2. Calculate Predicted Weight
                predicted_task_weight = actual_task_weight * prediction_noise
                
                # 3. Use PREDICTED weight for calculations
                # This simulates the algorithm making decisions based on imperfect information.
                # Every time fitness is checked, the "environment" fluctuates slightly.
                
                path_delay = get_path_delay(resource_bs_id, user_bs_id, task.data_size, data, user, graph)
                exe_delay = get_exe_delay(av_frequency, predicted_task_weight) 
                delay = path_delay + exe_delay
                
                # Energy = Power * Time (using the noisy time)
                power_watts = resource.power_model_parameters.get('static_power_percentage', 0)
                energy_consumption = power_watts * exe_delay
                
                total_energy += energy_consumption
                total_latency += delay
                
                # Check deadlines against the noisy realization
                if delay > task.deadline: individual.missed_deadlines += 1

        num_servers = data['EdgeServer'].count()
        individual.energy = total_energy / num_servers
        individual.latency = total_latency / num_servers
        individual.cost = total_cost 
        energy_values.append(individual.energy)
        latency_values.append(individual.latency)
        cost_values.append(individual.cost)

    min_e, max_e = min(energy_values), max(energy_values)
    min_l, max_l = min(latency_values), max(latency_values)
    min_c, max_c = min(cost_values), max(cost_values)

    for individual in population:
        norm_e = (individual.energy - min_e)/(max_e - min_e) if max_e > min_e else 0.5
        norm_l = (individual.latency - min_l)/(max_l - min_l) if max_l > min_l else 0.5
        norm_c = (individual.cost - min_c)/(max_c - min_c) if max_c > min_c else 0.5
        
        penalty = (individual.missed_deadlines * 1.0) + (100.0 if individual.mem_overload else 0.0)
        individual.fitness = [norm_e + penalty, norm_l + penalty, norm_c + penalty]

    return population