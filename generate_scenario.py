import os
import random
import argparse
from edge_sim_py import *
from edge_sim_py.dataset_generator import *
from edge_sim_py.components.mobility_models import random_mobility
from edge_sim_py.dataset_generator.edge_servers import raspberry_pi4, e5430

# --- Step 1: Parse Command-Line Arguments ---
parser = argparse.ArgumentParser(description="Generate a custom Fog scenario.")
parser.add_argument("--scenario_name", type=str, default="Base_Case")
parser.add_argument("--users", type=int, default=50)
parser.add_argument("--tier1", type=int, default=15)
parser.add_argument("--tier2", type=int, default=4)
parser.add_argument("--avg_weight", type=int, default=3)
parser.add_argument("--avg_data_size", type=int, default=500)
parser.add_argument("--deadline", type=float, default=30.0)
args = parser.parse_args()

# --- Step 2: Define Scenario Size ---
NUM_USERS = args.users
NUM_TIER1_NODES = args.tier1
NUM_TIER2_SERVERS = args.tier2
NUM_TIER3_CLOUD = 1 
TOTAL_SERVERS = NUM_TIER1_NODES + NUM_TIER2_SERVERS + NUM_TIER3_CLOUD
SCENARIO_FILENAME = f"{args.scenario_name}_ES-{TOTAL_SERVERS}_ED-{NUM_USERS}"

# Calculate Grid Size
TOTAL_BS_NEEDED = NUM_TIER1_NODES + NUM_TIER2_SERVERS + NUM_TIER3_CLOUD
MAP_SIZE = int(TOTAL_BS_NEEDED**0.5) + 2
TOTAL_BS = MAP_SIZE * MAP_SIZE

print(f"Generating: {SCENARIO_FILENAME} | Grid: {MAP_SIZE}x{MAP_SIZE}")

# --- Step 3: Output Dir ---
output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)

# --- Step 4: Map ---
map_coordinates = quadratic_grid(x_size=MAP_SIZE, y_size=MAP_SIZE)

# --- Step 5: Base Stations ---
base_stations = []
network_switches = []
for i, coords in enumerate(map_coordinates):
    bs = BaseStation()
    bs.id = i + 1
    bs.coordinates = coords
    bs.wireless_delay = 100 # 100 Mbps wireless link assumed
    base_stations.append(bs)
    
    switch = NetworkSwitch()
    switch.id = i + 1
    network_switches.append(switch)
    bs._connect_to_network_switch(switch)

# --- Step 6: Edge Servers ---
edge_servers = []
server_id_counter = 1

# Tier 1 (Pi - Fog Nodes)
for i in range(NUM_TIER1_NODES):
    fog_node = raspberry_pi4()
    fog_node.id = server_id_counter
    fog_node.power_model_parameters["monetary_cost"] = 1
    edge_servers.append(fog_node)
    
    # Distribute them across the map
    if i < len(base_stations):
        base_stations[i]._connect_to_edge_server(fog_node)
    server_id_counter += 1

# Tier 2 (Xeon - Edge Servers)
for i in range(NUM_TIER2_SERVERS):
    fog_server = e5430()
    fog_server.id = server_id_counter
    fog_server.power_model_parameters["monetary_cost"] = 3
    edge_servers.append(fog_server)
    
    # Place them after Tier 1 nodes
    idx = NUM_TIER1_NODES + i
    if idx < len(base_stations):
        base_stations[idx]._connect_to_edge_server(fog_server)
    server_id_counter += 1

# Tier 3 (Cloud)
cloud_server = EdgeServer()
cloud_server.id = server_id_counter
cloud_server.model_name = "Cloud-Server"
cloud_server.cpu = 64
cloud_server.memory = 262144
cloud_server.disk = 1048576
cloud_server.frequency = 3.2 * 1e9 # 3.2 GHz
cloud_server.power_model_parameters = {
    "max_power_consumption": 600,
    "static_power_percentage": 65,
    "monetary_cost": 10
}
edge_servers.append(cloud_server)

# Place Cloud at the "End" of the grid or middle
c_idx = NUM_TIER1_NODES + NUM_TIER2_SERVERS
if c_idx < len(base_stations):
    base_stations[c_idx]._connect_to_edge_server(cloud_server)

# --- Step 7: Network Topology (Backbone) ---
topology = Topology()
topology.id = 1
topology.add_nodes_from(network_switches)

# Create a Chain/Ring for local connectivity + Links to Cloud
cloud_switch = cloud_server.base_station.network_switch if cloud_server.base_station else network_switches[-1]

for i in range(len(network_switches)):
    # 1. Connect to next neighbor (Ring/Chain)
    if i < len(network_switches) - 1:
        next_switch = network_switches[i+1]
        link = NetworkLink()
        link.id = len(topology.edges) + 1
        link.nodes = [network_switches[i], next_switch]
        link.bandwidth = 1000 # 1 Gbps local
        link.delay = 0.005 # 5ms
        link.topology = topology
        topology.add_edge(link.nodes[0], link.nodes[1])
        # Register for Simulator
        if link.nodes[0] in topology._adj: topology._adj[link.nodes[0]][link.nodes[1]] = link
        if link.nodes[1] in topology._adj: topology._adj[link.nodes[1]][link.nodes[0]] = link

    # 2. Connect randomly to Cloud (Backbone)
    # Every 5th switch gets a direct uplink to Cloud (or all if small map)
    if i % 5 == 0 and network_switches[i] != cloud_switch:
        link = NetworkLink()
        link.id = len(topology.edges) + 1
        link.nodes = [network_switches[i], cloud_switch]
        link.bandwidth = 500 # 500 Mbps Backbone
        link.delay = 0.05 # 50ms Backbone Latency
        link.topology = topology
        topology.add_edge(link.nodes[0], link.nodes[1])
        if link.nodes[0] in topology._adj: topology._adj[link.nodes[0]][link.nodes[1]] = link
        if link.nodes[1] in topology._adj: topology._adj[link.nodes[1]][link.nodes[0]] = link

# --- Step 8: Users ---
for i in range(NUM_USERS):
    user = User()
    user.id = i + 1
    
    # Users prefer Tier 1 areas (closer to edge)
    valid_bs = base_stations[:NUM_TIER1_NODES + NUM_TIER2_SERVERS] 
    if not valid_bs: valid_bs = base_stations
    user_bs = random.choice(valid_bs)
    
    user.coordinates = user_bs.coordinates
    user.coordinates_trace = [user_bs.coordinates]
    user.base_station = user_bs
    user_bs.users.append(user)
    user.mobility_model = random_mobility
    
    app = Application()
    app.id = i + 1
    
    service = Service()
    service.id = i + 1
    service.cpu_demand = random.randint(50, 200)
    service.memory_demand = random.randint(32, 128)
    
    # Task Weight (Cycles)
    w_base = max(1, args.avg_weight)
    service.weight = random.randint(w_base, w_base + 3) * 1e9
    
    # Data Size (MB)
    d_base = max(50, args.avg_data_size)
    service.data_size = random.randint(d_base, d_base + 200)
    
    service.deadline = args.deadline
    
    app.connect_to_service(service)
    user._connect_to_application(app, delay_sla=service.deadline)

# --- Step 9: Export ---
print(f"Exporting scenario to {output_dir}/{SCENARIO_FILENAME}.json ...")
ComponentManager.export_scenario(save_to_file=True, file_name=SCENARIO_FILENAME)
print("Done!")