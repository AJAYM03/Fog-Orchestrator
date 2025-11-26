import os
import random
import argparse
from edge_sim_py import *
from edge_sim_py.dataset_generator import *
from edge_sim_py.components.mobility_models import random_mobility
from edge_sim_py.dataset_generator.edge_servers import raspberry_pi4, e5430

# --- Step 1: Parse Command-Line Arguments ---
parser = argparse.ArgumentParser(description="Generate a custom Fog scenario.")
parser.add_argument("--scenario_name", type=str, default="Base_Case", help="Name for the output folder/file prefix")
parser.add_argument("--users", type=int, default=50, help="Number of users (sensors) to create")
parser.add_argument("--tier1", type=int, default=15, help="Number of Tier 1 (Fog Node) servers")
parser.add_argument("--tier2", type=int, default=4, help="Number of Tier 2 (Fog Server) servers")
parser.add_argument("--avg_weight", type=int, default=3, help="Average compute weight (X * 10e9)")
parser.add_argument("--avg_data_size", type=int, default=500, help="Average data size (in MB)")
parser.add_argument("--deadline", type=float, default=30.0, help="Absolute deadline for all tasks (in seconds)")
args = parser.parse_args()

# --- Step 2: Define Scenario Size from Arguments ---
NUM_USERS = args.users
NUM_TIER1_NODES = args.tier1
NUM_TIER2_SERVERS = args.tier2
NUM_TIER3_CLOUD = 1   # We always have 1 Cloud
TOTAL_SERVERS = NUM_TIER1_NODES + NUM_TIER2_SERVERS + NUM_TIER3_CLOUD
SCENARIO_FILENAME = f"{args.scenario_name}_ES-{TOTAL_SERVERS}_ED-{NUM_USERS}"

# Make a grid big enough to hold all servers, plus a few empty spots
TOTAL_BS_NEEDED = NUM_TIER1_NODES + NUM_TIER2_SERVERS + NUM_TIER3_CLOUD
MAP_SIZE = int(TOTAL_BS_NEEDED**0.5) + 2
TOTAL_BS = MAP_SIZE * MAP_SIZE

print(f"Generating scenario: {SCENARIO_FILENAME}")
print(f"Grid size: {MAP_SIZE}x{MAP_SIZE} ({TOTAL_BS} Base Stations)")
print(f"Users: {NUM_USERS}, Servers: {TOTAL_SERVERS} (T1: {NUM_TIER1_NODES}, T2: {NUM_TIER2_SERVERS}, T3: {NUM_TIER3_CLOUD})")

# --- Step 3: Create directories ---
output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)

# --- Step 4: Define the map coordinates ---
map_coordinates = quadratic_grid(x_size=MAP_SIZE, y_size=MAP_SIZE)

# --- Step 5: Create Base Stations and Switches ---
base_stations = []
network_switches = []
for i, coords in enumerate(map_coordinates):
    bs = BaseStation()
    bs.id = i + 1
    bs.coordinates = coords
    bs.wireless_delay = 100 # Interpreted as 100 Mbps bandwidth by our config
    base_stations.append(bs)
    
    switch = NetworkSwitch()
    switch.id = i + 1
    network_switches.append(switch)
    bs._connect_to_network_switch(switch)

# --- Step 6: Create Edge Servers (Fog Tiers) ---
edge_servers = []
server_id_counter = 1

# Tier 1 (Fog Nodes)
for i in range(NUM_TIER1_NODES):
    fog_node = raspberry_pi4()
    fog_node.id = server_id_counter
    fog_node.power_model_parameters["monetary_cost"] = 1
    edge_servers.append(fog_node)
    base_stations[i]._connect_to_edge_server(fog_node) # Connect to first N base stations
    server_id_counter += 1

# Tier 2 (Fog Servers)
for i in range(NUM_TIER2_SERVERS):
    fog_server = e5430()
    fog_server.id = server_id_counter
    fog_server.power_model_parameters["monetary_cost"] = 3
    edge_servers.append(fog_server)
    base_stations[NUM_TIER1_NODES + i]._connect_to_edge_server(fog_server) # Connect to next batch
    server_id_counter += 1

# Tier 3 (Cloud Server)
cloud_server = EdgeServer()
cloud_server.id = server_id_counter
cloud_server.model_name = "Cloud-Server"
cloud_server.cpu = 1000
cloud_server.memory = 999999
cloud_server.disk = 999999
cloud_server.power_model_parameters = {
    "static_power_percentage": 200,
    "monetary_cost": 10
}
edge_servers.append(cloud_server)
base_stations[NUM_TIER1_NODES + NUM_TIER2_SERVERS]._connect_to_edge_server(cloud_server) # Connect to next

# --- Step 7: Create the Network Topology ---
topology = Topology()
topology.id = 1
topology.add_nodes_from(network_switches)

cloud_switch = cloud_server.base_station.network_switch

for i in range(len(network_switches)):
    for j in range(i + 1, len(network_switches)):
        link = NetworkLink()
        link.id = len(topology.edges) + 1
        link.nodes = [network_switches[i], network_switches[j]]
        link.topology = topology # Fix for the NoneType error

        if network_switches[i] == cloud_switch or network_switches[j] == cloud_switch:
            link.bandwidth = 100   # 100 Mbps (Slow)
            link.delay = 50        # 50ms (High Delay)
        else:
            link.bandwidth = 1000  # 1000 Mbps (Fast)
            link.delay = 5         # 5ms (Low Delay)
        
        topology.add_edge(link.nodes[0], link.nodes[1])
        topology._adj[link.nodes[0]][link.nodes[1]] = link
        topology._adj[link.nodes[1]][link.nodes[0]] = link

# --- Step 8: Create Users (Sensors) and their Tasks ---
for i in range(NUM_USERS):
    user = User()
    user.id = i + 1
    
    # Place user at a random base station (can't be the cloud one)
    user_bs = random.choice([bs for bs in base_stations if bs != cloud_server.base_station])
    user.coordinates = user_bs.coordinates
    user.coordinates_trace = [user_bs.coordinates, user_bs.coordinates] # Stationary user fix
    user.base_station = user_bs
    user_bs.users.append(user)
    user.mobility_model = random_mobility # Still needed by simulator
    
    app = Application()
    app.id = i + 1
    
    # Create a service (task) with parameters from args
    service = Service()
    service.id = i + 1
    service.cpu_demand = random.randint(50, 200)
    service.memory_demand = random.randint(32, 128)
    
    # Use the args to set a randomized range
    service.weight = random.randint(max(1, args.avg_weight - 1), args.avg_weight + 2) * 10e9
    service.data_size = random.randint(max(50, args.avg_data_size - 100), args.avg_data_size + 200)
    service.deadline = args.deadline
    
    app.connect_to_service(service)
    user._connect_to_application(app, delay_sla=100)

# --- Step 9: Export the scenario to JSON! ---
file_name_base = SCENARIO_FILENAME
full_file_path = f"{output_dir}/{file_name_base}.json"

print(f"Exporting scenario to {full_file_path} ...")
ComponentManager.export_scenario(
    save_to_file=True,
    file_name=file_name_base
)
print("Done!")