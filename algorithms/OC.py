from config import Individual
import random

class OC:
    def __init__(self, fitness, data):
        self.fitness = fitness
        self.data = data
        self.static_power_percentage = 0.000000001

    def calculate_delay(self, task):
        # Simulation of cloud parameters
        frequency = random.uniform(8 * 10**9, 11 * 10**9)
        bandwidth = random.uniform(4 * 10**3, 6 * 10**3)
        communication_delay = task['data_size'] / bandwidth
        static_network_latency = 3
        computation_delay = task['weight'] / frequency
        return communication_delay + computation_delay + static_network_latency, frequency
    
    def calculate_energy(self, task):
        # Simulation of cloud energy
        energy_consumption = task['weight'] * self.static_power_percentage
        return energy_consumption
    
    def schedule(self):
        individual = Individual()
        tasks = [user.applications[0].services[0] for user in self.data["User"].all()]
        total_delay = 0
        total_energy = 0
        missed_deadlines = 0
        max_delay = 0

        # 1. Calculate Metrics (Internal Simulation)
        for task in tasks:
            task_vars = vars(task)
            delay, frequency = self.calculate_delay(task_vars)
            energy = self.calculate_energy(task_vars)
            total_delay += delay
            total_energy += energy
            if delay > task_vars['deadline']:
                missed_deadlines += 1
            if delay > max_delay:
                max_delay = delay

        num_tasks = len(tasks)
        qos = (num_tasks - missed_deadlines) / num_tasks
        avg_latency = total_delay / num_tasks if num_tasks > 0 else 0
        
        individual.qos = qos
        individual.resource_utilization = 1
        individual.missed_deadlines = missed_deadlines
        individual.latency = avg_latency
        individual.energy = total_energy
        individual.max_resource_latency = max_delay
        individual.cost = 0 

        # 2. NEW: Generate Schedule Data (For Dashboard Visualization)
        # We must tell the dashboard that everyone is on the Cloud.
        
        # Find the Cloud Server's index in the list
        cloud_server_index = 0
        all_servers = self.data['EdgeServer'].all()
        for i, server in enumerate(all_servers):
            if "Cloud" in server.model_name:
                cloud_server_index = i
                break
        
        # Build the CInd list: All users -> Cloud Server
        # Format: [0, 0, 1, 0, 0...] repeated for each user
        num_servers = len(all_servers)
        c_ind = []
        for _ in range(num_tasks):
            user_gene = [0] * num_servers
            user_gene[cloud_server_index] = 1 # Set Cloud to 1
            c_ind.extend(user_gene)
            
        individual.CInd = c_ind

        return individual

    def run(self):
        return [self.schedule()]