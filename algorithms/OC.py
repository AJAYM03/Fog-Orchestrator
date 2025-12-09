from config import Individual

class OC:
    def __init__(self, fitness, data):
        self.fitness = fitness
        self.data = data

    def run(self):
        individual = Individual()
        individual.CInd = []
        edge_servers = self.data['EdgeServer'].all()
        cloud_index = 0
        for i, server in enumerate(edge_servers):
            if "Cloud" in server.model_name:
                cloud_index = i
                break
        num_tasks = self.data['User'].count()
        num_servers = len(edge_servers)
        for _ in range(num_tasks):
            gene = [0] * num_servers
            gene[cloud_index] = 1
            individual.CInd.extend(gene)
        
        population = [individual]
        population = self.fitness(population, self.data)
        return population